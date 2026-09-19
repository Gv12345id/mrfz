#!/usr/bin/env python
r"""阶段门禁入口：`python scripts/verify.py --phase N`。

本脚本是 PLANS.md 里所有停止条件的唯一执行入口，输出契约固定的 JSON：

```json
{"phase": 1, "passed": true, "started_at": "2026-09-19T10:00:00+08:00",
 "checks": [{"name": "screenshot-p95-ms", "passed": true, "detail": "",
             "value": 412.0, "threshold": 1500}]}
```

规则（PLANS.md「全局基础设施」）：

- `passed` 是所有 `checks[].passed` 的逻辑与；任一 check 失败则退出码非 0。
- `value` / `threshold` 写机器可比对的数字，不写文字结论。
- stdout 只有 JSON；人类可读摘要写 stderr；`--json-out` 落盘同一份 JSON。

真机类 check 在拿不到设备时**如实判 fail**（`detail` 写明原因），不 skip、
不放宽阈值：门禁宁可变红，也不给假绿灯。

用法：`.\.venv\Scripts\python.exe scripts\verify.py --phase 1 [--only a,b]`
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("arknights_agent.verify")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
SCHEMA_DOC = ROOT / "docs" / "specs" / "2026-09-19-mcp-tool-schema.md"
INTEGRATION_TEST = ROOT / "tests" / "integration" / "test_device_loop.py"
RUNS_DIR = ROOT / "runs"
COVERAGE_JSON = RUNS_DIR / "coverage.json"
DEVICE_LOOP_METRICS = RUNS_DIR / "phase1" / "device_loop.json"

# 阈值来自 PLANS.md 阶段 1 验收标准；改阈值必须同时改 PLANS.md 并说明理由。
MIN_COVERAGE_PERCENT = 80.0
SCREENSHOT_P95_THRESHOLD_MS = 1500.0
DEVICE_LOOP_FRAMES = 100
DEVICE_LOOP_CLICKS = 20

# 阶段 1 必须锁定的 MCP 工具（见 docs/specs/2026-09-19-mcp-tool-schema.md）。
REQUIRED_TOOLS = ("find_adb_device_list", "connect_adb_device", "screencap", "click", "swipe")
COVERAGE_PACKAGES = ("mcp", "device")

STATIC_TIMEOUT_S = 600.0
OFFLINE_TESTS_TIMEOUT_S = 1800.0
INTEGRATION_TIMEOUT_S = 1800.0
DOCTOR_TIMEOUT_S = 600.0
SCHEMA_PROBE_TIMEOUT_S = 240.0

LOCAL_TZ = timezone(timedelta(hours=8))
SHA256_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")
PYTEST_PASSED_PATTERN = re.compile(r"(\d+) passed")


@dataclass(frozen=True)
class CheckResult:
    """单个门禁检查的结果；`value` / `threshold` 不适用时为 None。"""

    name: str
    passed: bool
    detail: str = ""
    value: float | int | bool | str | None = None
    threshold: float | int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class CheckContext:
    """一次门禁运行共享的上下文（用于判断 metrics 是否本次产出）。"""

    started_ts: float


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #


def child_env() -> dict[str, str]:
    """子进程环境：把当前解释器所在目录（venv/Scripts）加进 PATH。

    `maa-mcp` 只装在 venv 里，未激活虚拟环境时 `arknights-agent doctor`
    会找不到它；这里显式补 PATH，保证用哪个解释器跑 verify.py 都能复现。
    """

    env = dict(os.environ)
    interpreter_dir = str(Path(sys.executable).parent)
    env["PATH"] = interpreter_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_command(
    command: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """在仓库根目录跑一个子进程并捕获输出；每个等待都有超时。"""

    return subprocess.run(
        list(command),
        cwd=str(ROOT),
        env=dict(env) if env is not None else child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )


def tail(text: str, *, limit: int = 600) -> str:
    """取输出的最后几行非空内容，用于 detail。"""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "<无输出>"
    return " | ".join(lines[-3:])[:limit]


def number_field(data: Mapping[str, Any], key: str) -> float | None:
    """取数值字段；缺失或非数值（bool 不算数值）时返回 None。"""

    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# --------------------------------------------------------------------------- #
# 可单测的纯函数
# --------------------------------------------------------------------------- #


def canonical_schema_hash(tools: Sequence[Mapping[str, Any]]) -> str:
    """MCP 工具 schema 的 canonical hash。

    只取 `name` 与 `inputSchema`（描述措辞会变，不进 hash），按工具名排序后
    做紧凑 JSON 编码，再取 sha256。
    """

    canonical = [
        {"name": str(tool["name"]), "inputSchema": tool["inputSchema"]}
        for tool in sorted(tools, key=lambda tool: str(tool["name"]))
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def aggregate_coverage(coverage: Mapping[str, Any], package: str) -> float | None:
    """聚合某个包（`arknights_agent/<package>/`）的语句覆盖率百分比。

    数据来自 `pytest --cov-report=json`。包里一个文件都没统计到时返回 None，
    由调用方判 fail，而不是当成 0% 或 100% 蒙混过去。
    """

    files = coverage.get("files")
    if not isinstance(files, Mapping):
        return None
    marker = f"/arknights_agent/{package}/"
    covered = 0
    statements = 0
    for raw_path, entry in files.items():
        normalized = "/" + Path(str(raw_path)).as_posix().lstrip("/")
        if marker not in normalized or not isinstance(entry, Mapping):
            continue
        summary = entry.get("summary")
        if not isinstance(summary, Mapping):
            continue
        covered += int(summary.get("covered_lines", 0))
        statements += int(summary.get("num_statements", 0))
    if statements == 0:
        return None
    return round(covered / statements * 100, 2)


def p95_ms(samples_ms: Sequence[float]) -> float:
    """最近秩法 p95：升序后取 `ceil(0.95 * n)` 位（n=100 时即第 95 位）。"""

    if not samples_ms:
        raise ValueError("样本为空，无法计算 p95")
    ordered = sorted(samples_ms)
    rank = min(len(ordered), max(1, math.ceil(0.95 * len(ordered))))
    return round(float(ordered[rank - 1]), 3)


def build_report(phase: int, checks: Sequence[CheckResult], started_at: datetime) -> dict[str, Any]:
    """按 PLANS.md 契约组装门禁报告。"""

    return {
        "phase": phase,
        "passed": all(check.passed for check in checks),
        "started_at": started_at.isoformat(),
        "checks": [check.to_json() for check in checks],
    }


def exit_code(report: Mapping[str, Any]) -> int:
    """`passed` 为真返回 0，否则非 0。"""

    return 0 if report.get("passed") else 1


def documented_schema_hash(doc_text: str) -> str | None:
    """从 schema 文档里取出记录的 sha256（没有则 None）。"""

    match = SHA256_PATTERN.search(doc_text)
    return match.group(0) if match else None


# --------------------------------------------------------------------------- #
# 重活：离线测试套件（含覆盖率）与线上 MaaMCP schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OfflineSuite:
    """一次 `pytest -m "not integration"` + 覆盖率运行的结果。"""

    passed: bool
    passed_count: int | None
    coverage: Mapping[str, float | None]
    detail: str


@functools.lru_cache(maxsize=1)
def offline_suite() -> OfflineSuite:
    """跑离线测试并产出 `runs/coverage.json`（进程内只跑一次）。"""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not integration",
        "-q",
        *(f"--cov=arknights_agent.{package}" for package in COVERAGE_PACKAGES),
        "--cov-report=term-missing",
        f"--cov-report=json:{COVERAGE_JSON}",
    ]
    COVERAGE_JSON.unlink(missing_ok=True)
    try:
        proc = run_command(command, timeout_s=OFFLINE_TESTS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return OfflineSuite(False, None, {}, f"离线测试超时（{OFFLINE_TESTS_TIMEOUT_S}s）")
    except OSError as exc:
        return OfflineSuite(False, None, {}, f"无法启动 pytest：{exc}")

    match = PYTEST_PASSED_PATTERN.search(proc.stdout)
    passed_count = int(match.group(1)) if match else None
    coverage: dict[str, float | None] = dict.fromkeys(COVERAGE_PACKAGES)
    if COVERAGE_JSON.is_file():
        try:
            data = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("覆盖率 JSON 无法解析：%s", exc)
        else:
            for package in COVERAGE_PACKAGES:
                coverage[package] = aggregate_coverage(data, package)

    detail = f"pytest -m 'not integration' 退出码 {proc.returncode}"
    if proc.returncode != 0:
        detail = f"{detail}：{tail(proc.stdout)}"
    return OfflineSuite(proc.returncode == 0, passed_count, coverage, detail)


@functools.lru_cache(maxsize=1)
def live_schema_tools() -> list[dict[str, Any]]:
    """用本仓库的 MCP 客户端拉一次线上 MaaMCP 的 `tools/list`。"""

    sys.path.insert(0, str(ROOT / "src"))
    from arknights_agent.config import load_settings
    from arknights_agent.mcp.client import McpStdioClient

    settings = load_settings(CONFIG_DIR)
    command = settings.mcp.command
    if shutil.which(command) is None:
        candidate = Path(sys.executable).parent / f"{command}.exe"
        if candidate.is_file():
            settings = settings.model_copy(
                update={"mcp": settings.mcp.model_copy(update={"command": str(candidate)})}
            )

    async def probe() -> list[dict[str, Any]]:
        async with McpStdioClient(settings.mcp) as client:
            listed = await asyncio.wait_for(
                client.session.list_tools(), timeout=settings.mcp.call_timeout_s
            )
        return [{"name": tool.name, "inputSchema": tool.inputSchema} for tool in listed.tools]

    async def probe_with_timeout() -> list[dict[str, Any]]:
        return await asyncio.wait_for(probe(), timeout=SCHEMA_PROBE_TIMEOUT_S)

    return asyncio.run(probe_with_timeout())


def read_device_loop_metrics(started_ts: float) -> tuple[Mapping[str, Any] | None, str]:
    """读取集成测试产出的 metrics；陈旧文件不算数。"""

    relative = DEVICE_LOOP_METRICS.relative_to(ROOT)
    if not DEVICE_LOOP_METRICS.is_file():
        return None, f"缺少 {relative}：真机集成测试尚未成功产出指标"
    try:
        data = json.loads(DEVICE_LOOP_METRICS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{relative} 无法解析：{exc}"
    if not isinstance(data, Mapping):
        return None, f"{relative} 顶层不是对象"
    if DEVICE_LOOP_METRICS.stat().st_mtime < started_ts:
        return None, f"{relative} 早于本次运行（陈旧文件），请连同 device-loop-tests 一起跑"
    return data, ""


# --------------------------------------------------------------------------- #
# 阶段 1 的 check
# --------------------------------------------------------------------------- #


def check_static_checks(_: CheckContext) -> CheckResult:
    """验收 1：ruff format --check / ruff check / mypy src 全绿。"""

    steps = (
        ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
        ("mypy src", [sys.executable, "-m", "mypy", "src"]),
    )
    failures: list[str] = []
    for label, command in steps:
        try:
            proc = run_command(command, timeout_s=STATIC_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            failures.append(f"{label} 超时（{STATIC_TIMEOUT_S}s）")
            continue
        except OSError as exc:
            failures.append(f"{label} 无法启动：{exc}")
            continue
        if proc.returncode != 0:
            failures.append(f"{label} 退出码 {proc.returncode}：{tail(proc.stdout + proc.stderr)}")
    if failures:
        return CheckResult("static-checks", False, "；".join(failures))
    return CheckResult("static-checks", True, "ruff format / ruff check / mypy src 全部通过")


def check_offline_tests(_: CheckContext) -> CheckResult:
    """验收 2：`pytest -m "not integration"` 全绿。"""

    suite = offline_suite()
    detail = f"{suite.passed_count} passed" if suite.passed_count is not None else suite.detail
    return CheckResult("offline-tests", suite.passed, detail, suite.passed_count, None)


def _coverage_check(package: str) -> CheckResult:
    name = f"coverage-{package}"
    suite = offline_suite()
    percent = suite.coverage.get(package)
    if percent is None:
        return CheckResult(
            name,
            False,
            f"未拿到 arknights_agent.{package} 的覆盖率数据（缺 pytest-cov 或没有统计到文件）",
            None,
            MIN_COVERAGE_PERCENT,
        )
    passed = percent >= MIN_COVERAGE_PERCENT
    return CheckResult(
        name,
        passed,
        f"arknights_agent.{package} 语句覆盖率 {percent}%",
        percent,
        MIN_COVERAGE_PERCENT,
    )


def check_coverage_mcp(_: CheckContext) -> CheckResult:
    """验收 2：`mcp/` 覆盖率 ≥ 80%。"""

    return _coverage_check("mcp")


def check_coverage_device(_: CheckContext) -> CheckResult:
    """验收 2：`device/` 覆盖率 ≥ 80%。"""

    return _coverage_check("device")


def check_schema_doc(_: CheckContext) -> CheckResult:
    """验收 6：schema 文档存在，且含工具名、参数表与 schema hash。"""

    relative = SCHEMA_DOC.relative_to(ROOT)
    if not SCHEMA_DOC.is_file():
        return CheckResult("schema-doc", False, f"缺少 {relative}")
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    missing = [tool for tool in REQUIRED_TOOLS if tool not in text]
    if documented_schema_hash(text) is None:
        missing.append("sha256 hash")
    if "inputSchema" not in text:
        missing.append("inputSchema 参数表")
    if missing:
        return CheckResult("schema-doc", False, f"{relative} 缺少：{', '.join(missing)}")
    return CheckResult("schema-doc", True, f"{relative} 含工具名、参数表与 schema hash")


def check_schema_hash(_: CheckContext) -> CheckResult:
    """验收 6：文档记录的 hash 与线上 MaaMCP 的 schema 一致。"""

    relative = SCHEMA_DOC.relative_to(ROOT)
    if not SCHEMA_DOC.is_file():
        return CheckResult("schema-hash", False, f"缺少 {relative}，无法比对 hash")
    expected = documented_schema_hash(SCHEMA_DOC.read_text(encoding="utf-8"))
    if expected is None:
        return CheckResult("schema-hash", False, f"{relative} 里没有 sha256")
    try:
        tools = live_schema_tools()
    except Exception as exc:  # 起不来 server、超时、协议错误都算 fail
        return CheckResult(
            "schema-hash", False, f"拉取线上 MaaMCP schema 失败：{type(exc).__name__}: {exc}"
        )
    actual = canonical_schema_hash(tools)
    detail = f"线上 {len(tools)} 个工具，schema_sha256={actual}"
    if actual != expected:
        return CheckResult(
            "schema-hash",
            False,
            f"{detail}，与 {relative} 记录的 {expected} 不一致（升级 MaaMCP 后要同步文档）",
        )
    return CheckResult("schema-hash", True, f"{detail}，与文档一致")


def check_doctor_json(_: CheckContext) -> CheckResult:
    """验收 4：`arknights-agent doctor --json` 退出码 0 且字段达标。"""

    command = [
        sys.executable,
        "-m",
        "arknights_agent",
        "doctor",
        "--json",
        "--frames",
        str(DEVICE_LOOP_FRAMES),
    ]
    try:
        proc = run_command(command, timeout_s=DOCTOR_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return CheckResult("doctor-json", False, f"doctor 超时（{DOCTOR_TIMEOUT_S}s）")
    except OSError as exc:
        return CheckResult("doctor-json", False, f"doctor 无法启动：{exc}")

    if proc.returncode != 0:
        return CheckResult(
            "doctor-json",
            False,
            f"doctor 退出码 {proc.returncode}：{tail(proc.stdout + proc.stderr)}",
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return CheckResult("doctor-json", False, f"doctor --json 输出不是 JSON：{exc}")
    if not isinstance(report, Mapping):
        return CheckResult("doctor-json", False, "doctor --json 输出顶层不是对象")

    problems = [
        f"{key}={report.get(key)!r}" for key in ("mcp", "maa", "device") if report.get(key) != "ok"
    ]
    if report.get("resolution") != "1280x720":
        problems.append(f"resolution={report.get('resolution')!r}")
    p95 = number_field(report, "screenshot_p95_ms")
    if p95 is None:
        problems.append("screenshot_p95_ms 缺失或非数值")
    if problems:
        return CheckResult("doctor-json", False, "doctor 字段不达标：" + ", ".join(problems))
    return CheckResult("doctor-json", True, f"doctor 全字段达标（p95={p95}ms）")


def check_device_loop_tests(_: CheckContext) -> CheckResult:
    """验收 3：真机集成测试（100 帧 + 20 次点击闭环）全绿。"""

    DEVICE_LOOP_METRICS.unlink(missing_ok=True)
    relative = INTEGRATION_TEST.relative_to(ROOT)
    if not INTEGRATION_TEST.is_file():
        return CheckResult("device-loop-tests", False, f"缺少 {relative}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "integration",
        str(relative),
        "-q",
        "-rs",
    ]
    try:
        proc = run_command(command, timeout_s=INTEGRATION_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "device-loop-tests", False, f"真机集成测试超时（{INTEGRATION_TIMEOUT_S}s）"
        )
    except OSError as exc:
        return CheckResult("device-loop-tests", False, f"无法启动 pytest：{exc}")
    if proc.returncode != 0:
        return CheckResult(
            "device-loop-tests",
            False,
            f"pytest -m integration 退出码 {proc.returncode}：{tail(proc.stdout)}",
        )
    return CheckResult("device-loop-tests", True, tail(proc.stdout, limit=200))


def check_screenshot_failures(context: CheckContext) -> CheckResult:
    """验收 3：100 帧截图 0 失败。"""

    name = "screenshot-failures"
    data, problem = read_device_loop_metrics(context.started_ts)
    if data is None:
        return CheckResult(name, False, problem, None, 0)
    failures = number_field(data, "frame_failures")
    frames = number_field(data, "frames")
    if failures is None or frames is None:
        return CheckResult(name, False, "metrics 缺少 frames / frame_failures", None, 0)
    if frames != DEVICE_LOOP_FRAMES:
        return CheckResult(
            name, False, f"截图帧数 {frames:g}，要求 {DEVICE_LOOP_FRAMES}", failures, 0
        )
    return CheckResult(
        name, failures <= 0, f"{int(frames)} 帧截图，失败 {int(failures)} 帧", failures, 0
    )


def check_screenshot_p95(context: CheckContext) -> CheckResult:
    """验收 5：截图 p95 ≤ 1500ms（100 帧实测）。"""

    name = "screenshot-p95-ms"
    data, problem = read_device_loop_metrics(context.started_ts)
    if data is None:
        return CheckResult(name, False, problem, None, SCREENSHOT_P95_THRESHOLD_MS)
    p95 = number_field(data, "p95_ms")
    if p95 is None:
        return CheckResult(name, False, "metrics 缺少 p95_ms", None, SCREENSHOT_P95_THRESHOLD_MS)
    return CheckResult(
        name,
        p95 <= SCREENSHOT_P95_THRESHOLD_MS,
        f"100 帧截图 p95={p95}ms",
        p95,
        SCREENSHOT_P95_THRESHOLD_MS,
    )


def check_click_loop(context: CheckContext) -> CheckResult:
    """验收 3：20 次点击闭环 20/20。"""

    name = "click-loop-successes"
    data, problem = read_device_loop_metrics(context.started_ts)
    if data is None:
        return CheckResult(name, False, problem, None, DEVICE_LOOP_CLICKS)
    confirmed = number_field(data, "clicks_confirmed")
    attempted = number_field(data, "clicks_attempted")
    if confirmed is None or attempted is None:
        return CheckResult(
            name,
            False,
            "metrics 缺少 clicks_attempted / clicks_confirmed",
            None,
            DEVICE_LOOP_CLICKS,
        )
    detail = (
        f"点击闭环 {int(confirmed)}/{int(attempted)}，"
        f"要求 {DEVICE_LOOP_CLICKS}/{DEVICE_LOOP_CLICKS}"
    )
    passed = attempted == DEVICE_LOOP_CLICKS and confirmed >= DEVICE_LOOP_CLICKS
    return CheckResult(name, passed, detail, confirmed, DEVICE_LOOP_CLICKS)


PHASE_CHECKS: dict[int, tuple[tuple[str, Callable[[CheckContext], CheckResult]], ...]] = {
    1: (
        ("static-checks", check_static_checks),
        ("offline-tests", check_offline_tests),
        ("coverage-mcp", check_coverage_mcp),
        ("coverage-device", check_coverage_device),
        ("schema-doc", check_schema_doc),
        ("schema-hash", check_schema_hash),
        ("doctor-json", check_doctor_json),
        ("device-loop-tests", check_device_loop_tests),
        ("screenshot-failures", check_screenshot_failures),
        ("screenshot-p95-ms", check_screenshot_p95),
        ("click-loop-successes", check_click_loop),
    ),
}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="按 PLANS.md 的验收标准跑阶段门禁，输出契约 JSON。",
    )
    parser.add_argument("--phase", type=int, default=1, help="阶段号（已实现：1）")
    parser.add_argument(
        "--only", default=None, help="只跑指定 check（逗号分隔）；默认跑该阶段全部 check"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="报告落盘路径，默认 runs/verify/phase-<N>.json"
    )
    parser.add_argument("--verbose", action="store_true", help="把失败 check 的详情打到 stderr")
    return parser.parse_args(argv)


def run_checks(phase: int, selected: Sequence[str], started_ts: float) -> list[CheckResult]:
    context = CheckContext(started_ts=started_ts)
    results: list[CheckResult] = []
    for name, runner in PHASE_CHECKS[phase]:
        if name not in selected:
            continue
        try:
            result = runner(context)
        except Exception as exc:  # 门禁要把任何异常变成 fail，而不是崩掉
            LOGGER.exception("check %s 抛出异常", name)
            result = CheckResult(name, False, f"check 抛出异常：{type(exc).__name__}: {exc}")
        results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    # stdout 要能被 CI 无条件当 UTF-8 JSON 读；Windows 默认按 ANSI 码页写，
    # 中文 detail 会变成别人读不了的字节，所以这里显式钉死 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="[verify] %(message)s", stream=sys.stderr)
    args = parse_args(argv)
    started_ts = time.time()
    started_at = datetime.now(LOCAL_TZ)

    if args.phase not in PHASE_CHECKS:
        print(
            f"阶段 {args.phase} 的门禁尚未实现（已实现：{sorted(PHASE_CHECKS)}）；"
            "阶段 2-4 的 check 落地后在此注册。",
            file=sys.stderr,
        )
        return 2

    available = [name for name, _ in PHASE_CHECKS[args.phase]]
    if args.only:
        selected = [item.strip() for item in args.only.split(",") if item.strip()]
        unknown = [name for name in selected if name not in available]
        if unknown:
            print(
                f"未知的 check：{', '.join(unknown)}；可用：{', '.join(available)}", file=sys.stderr
            )
            return 2
    else:
        selected = available

    results = run_checks(args.phase, selected, started_ts)
    report = build_report(args.phase, results, started_at)

    destination = args.json_out or (RUNS_DIR / "verify" / f"phase-{args.phase}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    summary = " ".join(f"{check.name}={'ok' if check.passed else 'FAIL'}" for check in results)
    state = "通过" if report["passed"] else "未通过"
    print(f"阶段 {args.phase} 门禁：{state}｜{summary}", file=sys.stderr)
    print(f"报告：{destination}", file=sys.stderr)
    for check in results:
        if not check.passed:
            print(f"  - {check.name}: {check.detail}", file=sys.stderr)
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())

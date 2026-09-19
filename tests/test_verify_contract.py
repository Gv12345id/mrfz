"""`scripts/verify.py` 的 JSON 契约与聚合逻辑测试。

verify.py 是阶段门禁的唯一放行判据，CI 与阶段 4 的循环都只读它的 JSON，
所以契约本身要有测试兜住：字段名、`passed` 的语义、退出码、覆盖率聚合、
schema hash 的稳定性。
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

VERIFY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify.py"


def _load_verify() -> Any:
    spec = importlib.util.spec_from_file_location("verify_script", VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses 需要模块已注册在 sys.modules 里才能解析 cls.__module__。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()

EXPECTED_PHASE1_CHECKS = (
    "static-checks",
    "offline-tests",
    "coverage-mcp",
    "coverage-device",
    "schema-doc",
    "schema-hash",
    "doctor-json",
    "device-loop-tests",
    "screenshot-failures",
    "screenshot-p95-ms",
    "click-loop-successes",
)


def test_phase1_check_names_match_plans_acceptance_criteria() -> None:
    """阶段 1 的 check 名一旦改动，PLANS.md 的映射表必须同步。"""

    assert tuple(name for name, _ in verify.PHASE_CHECKS[1]) == EXPECTED_PHASE1_CHECKS


def test_report_matches_plans_json_contract() -> None:
    started_at = datetime(2026, 9, 19, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    checks = [
        verify.CheckResult("static-checks", True, "ok"),
        verify.CheckResult("screenshot-p95-ms", True, "", 412.0, 1500),
    ]

    report = verify.build_report(1, checks, started_at)

    assert set(report) == {"phase", "passed", "started_at", "checks"}
    assert report["phase"] == 1
    assert report["passed"] is True
    assert report["started_at"] == "2026-09-19T10:00:00+08:00"
    assert set(report["checks"][0]) == {"name", "passed", "detail", "value", "threshold"}
    assert report["checks"][0]["value"] is None
    assert report["checks"][1]["value"] == 412.0
    assert report["checks"][1]["threshold"] == 1500
    assert json.loads(json.dumps(report)) == report


def test_passed_is_logical_and_of_checks() -> None:
    started_at = datetime.now(UTC)
    ok = verify.CheckResult("a", True)
    failed = verify.CheckResult("b", False, "boom")

    assert verify.build_report(1, [ok, ok], started_at)["passed"] is True
    assert verify.build_report(1, [ok, failed], started_at)["passed"] is False


def test_exit_code_is_nonzero_when_any_check_fails() -> None:
    started_at = datetime.now(UTC)
    passing = verify.build_report(1, [verify.CheckResult("a", True)], started_at)
    failing = verify.build_report(1, [verify.CheckResult("a", False)], started_at)

    assert verify.exit_code(passing) == 0
    assert verify.exit_code(failing) != 0


def test_unimplemented_phase_and_unknown_check_exit_with_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """阶段 2-4 未实现、check 名写错时不能假装成功。"""

    assert verify.main(["--phase", "4"]) == 2
    capsys.readouterr()

    assert verify.main(["--phase", "1", "--only", "not-a-check"]) == 2
    assert "not-a-check" in capsys.readouterr().err


def test_aggregate_coverage_sums_covered_over_statements() -> None:
    coverage = {
        "files": {
            "src\\arknights_agent\\mcp\\client.py": {
                "summary": {"covered_lines": 80, "num_statements": 100}
            },
            "src/arknights_agent/mcp/tools.py": {
                "summary": {"covered_lines": 20, "num_statements": 100}
            },
            "src/arknights_agent/device/fake.py": {
                "summary": {"covered_lines": 9, "num_statements": 10}
            },
        }
    }

    assert verify.aggregate_coverage(coverage, "mcp") == 50.0
    assert verify.aggregate_coverage(coverage, "device") == 90.0


def test_aggregate_coverage_without_matching_files_is_none() -> None:
    assert verify.aggregate_coverage({"files": {}}, "mcp") is None
    assert verify.aggregate_coverage({}, "device") is None


def test_canonical_schema_hash_ignores_order_and_tracks_schema_changes() -> None:
    tools = [
        {
            "name": "click",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
        {"name": "screencap", "inputSchema": {"type": "object", "properties": {}}},
    ]
    reordered = list(reversed(tools))
    changed = [
        tools[0],
        {"name": "screencap", "inputSchema": {"type": "object", "properties": {"zoom": {}}}},
    ]

    digest = verify.canonical_schema_hash(tools)

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert verify.canonical_schema_hash(reordered) == digest
    assert verify.canonical_schema_hash(changed) != digest


def test_documented_schema_hash_reads_hash_from_doc() -> None:
    text = "schema_sha256: 144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a"

    assert (
        verify.documented_schema_hash(text)
        == "144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a"
    )
    assert verify.documented_schema_hash("没有 hash 的文档") is None


def test_p95_uses_nearest_rank() -> None:
    assert verify.p95_ms([float(value) for value in range(1, 101)]) == 95.0
    assert verify.p95_ms([7.5]) == 7.5
    assert verify.p95_ms([3.0, 1.0, 2.0]) == 3.0
    with pytest.raises(ValueError, match="样本为空"):
        verify.p95_ms([])


def test_number_field_rejects_bool_and_missing() -> None:
    assert verify.number_field({"p95_ms": 12.5}, "p95_ms") == 12.5
    assert verify.number_field({"ok": True}, "ok") is None
    assert verify.number_field({}, "p95_ms") is None

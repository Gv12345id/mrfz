"""真机集成测试：100 帧截图基准 + 20 次点击闭环（PLANS.md 阶段 1 验收 3/5）。

产出 ``runs/phase1/device_loop.json``，字段名与 ``scripts/verify.py`` 的
``screenshot-failures`` / ``screenshot-p95-ms`` / ``click-loop-successes``
三个 check 一一对应。

安全闸门写死在代码里，不靠提示词：

1. ``config.local.yaml`` 的 ``integration.confirm_screen`` 必须为 ``true``；
2. 点击前用录制的**主界面**参考帧确认当前画面就是主界面，不是就一次都不点；
3. 点击序列固定为「A/B 交替」，每次都要观察到目标界面才记一次确认；
4. 设备不是 16:9 时直接失败，不做换算、不猜测。
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from arknights_agent.config import Settings, load_settings
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import Frame, diff_ratio, read_bgr, size_of
from arknights_agent.mcp.client import McpStdioClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "tests" / "fixtures" / "device"
HOME_REFERENCE = REFERENCE_DIR / "home-1280x720.png"
LIST_REFERENCE = REFERENCE_DIR / "operator-list-1280x720.png"
METRICS_PATH = ROOT / "runs" / "phase1" / "device_loop.json"

FRAME_COUNT = 100
CLICK_COUNT = 20
P95_THRESHOLD_MS = 1500.0
LOGICAL_SIZE = (1280, 720)
TARGET_ASPECT_RATIO = 16 / 9

# 主界面带基建与看板娘动画：同一画面自比实测 0.01~0.07，而主界面与干员列表
# 之间是 0.89。0.25 落在中间且两侧都留足余量，既不会把主界面误判成干员列表，
# 也不会因动画误报。
HOME_MATCH_MAX_DIFF = 0.25
# 目标界面与另一界面参考帧之差至少要有这么大，才认定画面真的切过去了。
MIN_SCREEN_DIFF = 0.05
# 点击后等界面动画收敛再截图。
CLICK_SETTLE_S = 1.2

SAFETY_NOTE = (
    "全程只在主界面与干员列表之间导航：未进入战斗/生息演算，未消耗理智或材料，"
    "未点任何确认/升级/购买按钮；点击前已用录制参考帧确认起点为主界面"
)


def p95_ms(samples_ms: list[float]) -> float:
    """最近秩法 p95，与 ``scripts/verify.py`` 的 ``p95_ms`` 同一规则。"""

    ordered = sorted(samples_ms)
    rank = min(len(ordered), max(1, math.ceil(0.95 * len(ordered))))
    return round(float(ordered[rank - 1]), 3)


def resolve_screen(
    frame: Frame, home_reference: Frame, list_reference: Frame
) -> tuple[str, float, float]:
    """按「离哪张参考帧更近」判断当前界面，返回 (界面名, 与主界面之差, 与列表之差)。"""

    home_diff = diff_ratio(frame, home_reference)
    list_diff = diff_ratio(frame, list_reference)
    return ("home" if home_diff < list_diff else "operator_list"), home_diff, list_diff


def write_metrics(metrics: dict[str, Any]) -> None:
    """落盘 metrics（失败路径也要写，避免 verify.py 读到陈旧文件）。"""

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


async def run_device_loop(settings: Settings, metrics: dict[str, Any]) -> None:
    """跑完 100 帧截图基准与 20 次点击闭环，结果写进 ``metrics``。"""

    home_reference = read_bgr(HOME_REFERENCE)
    list_reference = read_bgr(LIST_REFERENCE)

    async with McpStdioClient(settings.mcp) as client:
        device = await MaaMcpDevice.connect(client, settings.device)
        metrics["device"] = device.name

        # 安全闸门 4：非 16:9 直接失败，不做换算也不猜测。
        native_width, native_height = device.native_size
        aspect = native_width / native_height
        if abs(aspect - TARGET_ASPECT_RATIO) > 0.01:
            pytest.fail(f"设备原生分辨率 {native_width}x{native_height} 不是 16:9，拒绝继续")

        # ---- 阶段 A：100 帧截图基准 ----
        samples: list[float] = []
        intervals: list[float] = []
        errors: list[str] = []
        previous_started: float | None = None
        for _ in range(FRAME_COUNT):
            started = time.perf_counter()
            if previous_started is not None:
                intervals.append((started - previous_started) * 1000)
            previous_started = started
            try:
                frame = await device.screenshot()
            except Exception as exc:  # 单帧失败要计数，不能整段中断
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            samples.append((time.perf_counter() - started) * 1000)
            if size_of(frame) != LOGICAL_SIZE:
                pytest.fail(f"第 {len(samples)} 帧尺寸 {size_of(frame)} 不是 1280x720")

        metrics["frames"] = len(samples)
        metrics["frame_failures"] = len(errors)
        metrics["frame_errors"] = errors[:5]
        metrics["p95_ms"] = p95_ms(samples) if samples else None
        metrics["interval_p95_ms"] = p95_ms(intervals) if intervals else None
        metrics["resolution"] = f"{LOGICAL_SIZE[0]}x{LOGICAL_SIZE[1]}"
        metrics["native_size"] = f"{native_width}x{native_height}"

        if len(samples) != FRAME_COUNT:
            pytest.fail(f"100 帧截图只成功 {len(samples)} 帧，失败 {len(errors)} 帧")
        if metrics["p95_ms"] > P95_THRESHOLD_MS:
            pytest.fail(f"截图 p95={metrics['p95_ms']}ms 超过阈值 {P95_THRESHOLD_MS}ms")

        # ---- 阶段 B：安全闸门 ----
        if not settings.integration.confirm_screen:
            pytest.fail(
                "integration.confirm_screen 不是 true：未人工确认游戏处于中立界面，已跳过全部点击"
            )
        tap_a = settings.integration.tap_a
        tap_b = settings.integration.tap_b
        if tap_a is None or tap_b is None:
            pytest.fail("integration.tap_a / tap_b 未配置，已跳过全部点击")

        # 安全闸门 2：起点必须是主界面，否则一次都不点。
        current = await device.screenshot()
        screen, home_diff, list_diff = resolve_screen(current, home_reference, list_reference)
        metrics["preflight"] = {
            "observed_screen": screen,
            "diff_vs_home": round(home_diff, 4),
            "diff_vs_list": round(list_diff, 4),
        }
        if screen != "home" or home_diff >= HOME_MATCH_MAX_DIFF:
            pytest.fail(
                f"点击前置检查未通过：当前画面判定为 {screen}"
                f"（与主界面差 {home_diff:.4f}，与干员列表差 {list_diff:.4f}），"
                "已跳过全部点击；请人工把游戏退回主界面后重跑"
            )

        # ---- 阶段 C：20 次点击闭环（A/B 交替 10 轮） ----
        per_click: list[dict[str, Any]] = []
        confirmed = 0
        for index in range(1, CLICK_COUNT + 1):
            opens_list = index % 2 == 1
            target = tap_a if opens_list else tap_b
            expected = "operator_list" if opens_list else "home"
            accepted = await device.tap(*target)
            await asyncio.sleep(CLICK_SETTLE_S)
            current = await device.screenshot()
            screen, home_diff, list_diff = resolve_screen(current, home_reference, list_reference)
            ok = (
                bool(accepted)
                and screen == expected
                and max(home_diff, list_diff) > MIN_SCREEN_DIFF
            )
            confirmed += int(ok)
            per_click.append(
                {
                    "click": index,
                    "target": f"{'干员入口' if opens_list else '返回'}{target}",
                    "expected_screen": expected,
                    "observed_screen": screen,
                    "diff_vs_home": round(home_diff, 4),
                    "diff_vs_list": round(list_diff, 4),
                    "tap_accepted": bool(accepted),
                    "confirmed": ok,
                }
            )

        metrics["clicks_attempted"] = len(per_click)
        metrics["clicks_confirmed"] = confirmed
        metrics["per_click"] = per_click
        if confirmed != CLICK_COUNT:
            pytest.fail(f"点击闭环只确认 {confirmed}/{CLICK_COUNT} 次，详见 metrics 的 per_click")


def test_hundred_frames_and_twenty_click_loop() -> None:
    """100 帧截图 0 失败且 p95 ≤ 1500ms；20 次「点击已知元素 → 画面变化」全部确认。"""

    metrics: dict[str, Any] = {
        "artifact": "device_loop",
        "measured_at": datetime.now().astimezone().isoformat(),
        "frames": 0,
        "frame_failures": 0,
        "p95_ms": None,
        "clicks_attempted": 0,
        "clicks_confirmed": 0,
        "per_click": [],
        "criterion": (
            "每次点击后截图；该帧必须离目标界面参考帧更近、且两张参考帧之差 > 0.05，"
            "同时 tap 返回 true 才算确认"
        ),
        "safety": SAFETY_NOTE,
        "reference_frames": {
            "home": str(HOME_REFERENCE.relative_to(ROOT)),
            "operator_list": str(LIST_REFERENCE.relative_to(ROOT)),
        },
    }
    try:
        asyncio.run(run_device_loop(load_settings(ROOT / "config"), metrics))
    finally:
        write_metrics(metrics)

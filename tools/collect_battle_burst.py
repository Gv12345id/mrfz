"""等画面进入战斗后立刻快采一批帧（阶段 2 采集用）。

为什么需要它：手工喊"好了"的时候游戏可能还在编队页/加载页（实测踩过一次，整批 450 帧
全是 Loading 画面被过滤掉）。本工具先轮询画面直到**真的进入战斗**，再开始采集：

- 战斗判定：与已知战斗参考帧差异 < 0.5、与主界面参考帧差异 > 0.5、平均亮度落在 55–145；
- 判定通过后，用**这张新鲜战斗帧**当过滤参考（而不是旧战斗帧，避免镜头平移误杀）；
- 采集用 interval=0 + 硬时限，采完直接写 annotation / jsonl / sha256。

用法::

    .\\.venv\\Scripts\\python.exe tools\\collect_battle_burst.py --name annotation_400e
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from arknights_agent.config import load_settings
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import diff_ratio, write_bgr
from arknights_agent.mcp.client import McpStdioClient

LOCAL_TZ = timezone(timedelta(hours=8))
COLLECT_PATH = Path(__file__).resolve().parent / "collect_frames.py"

BATTLE_REF = Path("runs/phase2/frames/annotation_battle/frame_0151.png")
HOME_REF = Path("tests/fixtures/device/home-1280x720.png")
BATTLE_MATCH_MAX = 0.50  # 与已知战斗帧的最大差异
HOME_MATCH_MIN = 0.50  # 与主界面的最小差异
MIN_BRIGHTNESS = 55.0
MAX_BRIGHTNESS = 145.0


def _load_collect() -> Any:
    spec = importlib.util.spec_from_file_location("collect_frames", COLLECT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def looks_like_battle(
    frame: np.ndarray, battle_ref: np.ndarray, home_ref: np.ndarray
) -> tuple[bool, dict[str, float]]:
    """判断这帧是不是战斗画面，并返回判定用到的数值（便于写日志）。"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    info = {
        "brightness": float(gray.mean()),
        "diff_vs_battle": diff_ratio(battle_ref, frame),
        "diff_vs_home": diff_ratio(home_ref, frame),
    }
    ok = (
        info["diff_vs_battle"] < BATTLE_MATCH_MAX
        and info["diff_vs_home"] > HOME_MATCH_MIN
        and MIN_BRIGHTNESS <= info["brightness"] <= MAX_BRIGHTNESS
    )
    return ok, info


async def wait_for_battle(
    device: MaaMcpDevice, *, wait_seconds: float, poll_seconds: float
) -> np.ndarray:
    """轮询到战斗画面，返回那一帧；超时抛 TimeoutError。"""

    battle_ref = cv2.imread(str(BATTLE_REF))
    home_ref = cv2.imread(str(HOME_REF))
    if battle_ref is None or home_ref is None:
        raise FileNotFoundError("缺少战斗/主界面参考帧，无法判定画面")
    deadline = time.perf_counter() + wait_seconds
    while True:
        frame = await device.screenshot()
        ok, info = looks_like_battle(frame, battle_ref, home_ref)
        stamp = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
        print(
            f"[{stamp}] 亮度={info['brightness']:.0f} vs战斗={info['diff_vs_battle']:.2f} "
            f"vs主界面={info['diff_vs_home']:.2f} -> {'战斗画面 ✅' if ok else '等待中'}"
        )
        if ok:
            return frame
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"等待 {wait_seconds:.0f}s 仍未出现战斗画面")
        await asyncio.sleep(poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="等战斗画面出现后快采一批帧")
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--interval-ms", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--wait-seconds", type=float, default=120.0, help="等战斗画面的上限")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--name", default="annotation_battle_burst")
    parser.add_argument("--reference-threshold", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collect_module = _load_collect()
    started = time.perf_counter()

    async def run() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        settings = load_settings()
        async with McpStdioClient(settings.mcp) as client:
            device = await MaaMcpDevice.connect(client, settings.device)
            print(f"设备已连接：{device.name}，等待战斗画面…")
            frame = await wait_for_battle(
                device, wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds
            )
            reference_path = args.out_dir / "battle-reference.png"
            write_bgr(reference_path, frame)
            waited = time.perf_counter() - started
            print(f"检测到战斗画面（等待 {waited:.0f}s），开始采集 {args.frames} 帧…")
            records, meta = await collect_module.collect(
                frames=args.frames,
                interval_ms=args.interval_ms,
                out_dir=args.out_dir,
                screen_context="battle",
                frames_dir=args.out_dir / "frames" / args.name,
                reference_frame=reference_path,
                reference_threshold=args.reference_threshold,
                min_mean_brightness=40.0,
                max_seconds=args.max_seconds,
            )
        return records, meta, str(reference_path)

    try:
        records, meta, reference = asyncio.run(run())
    except TimeoutError as exc:
        print(f"失败：{exc}（请确认游戏已经在战斗里，而不是编队/加载页面）")
        return 1
    json_path, jsonl_path, sha_path = collect_module.write_annotation(
        records, meta, out_dir=args.out_dir, name=args.name
    )
    print(
        f"采集完成：保留 {meta['kept']}｜跳过 {meta['skipped']}｜失败 {meta['failures']}｜"
        f"参考帧 {reference}"
    )
    print(f"已写出：{json_path}｜{jsonl_path}｜{sha_path}")
    return 1 if meta["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())

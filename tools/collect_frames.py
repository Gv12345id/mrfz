"""阶段 2 只读采集器：连设备、抓 N 帧、落盘并生成标注骨架。

**绝不点击**：本脚本只调用 `MaaMcpDevice.screenshot()`（内部走 MCP 的 `screencap`），
不调用 click / swipe / click_key / input_text 中的任何一个。用它采集训练数据时，
游戏状态必须由人先摆好（例如进入战斗并让敌人出现在画面里）。

产出：

- `runs/phase2/frames/frame_XXXX.png`：逐帧 PNG（1280×720）；
- `runs/phase2/annotation_<N>.json`：带元信息的标注骨架，逐帧记录字段与阶段 1 的
  `player.jsonl` 对齐（`step / action / mcp_tool / params / result / elapsed_ms / screenshot`），
  另加 `resolution / sha256 / captured_at / labels`；
- `runs/phase2/annotation_<N>.jsonl`：同样的逐帧记录，一行一条，方便流式消费；
- `runs/phase2/frames.sha256`：帧文件校验和（PLANS.md 阶段 2 验收 6 用）。

用法::

    .\\.venv\\Scripts\\python.exe tools\\collect_frames.py --frames 100 --interval-ms 500 \\
        --screen-context main --out-dir runs/phase2 --name annotation_100
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from arknights_agent.config import load_settings
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import Frame, size_of, write_bgr
from arknights_agent.mcp.client import McpStdioClient

LOCAL_TZ = timezone(timedelta(hours=8))
DEFAULT_INTERVAL_MS = 500
MAX_CONSECUTIVE_FAILURES = 3
SCREEN_CONTEXTS = ("main", "battle", "other", "unknown")
ACTION_NAME = "SCREENCAP"
MCP_TOOL_NAME = "screencap"


def frame_record(
    *,
    step: int,
    screenshot: str,
    resolution: str,
    sha256: str,
    captured_at: datetime,
    elapsed_ms: float,
    controller_id: str,
    screen_context: str,
    result: str = "ok",
) -> dict[str, Any]:
    """构造一条逐帧记录。

    前 7 个字段与阶段 1 `player.jsonl` 的逐帧记录同名同义，后面是标注阶段额外需要的
    溯源信息；`labels` 由标注环节填充（采集时为空列表）。
    """

    return {
        "step": step,
        "action": ACTION_NAME,
        "mcp_tool": MCP_TOOL_NAME,
        "params": {"controller_id": controller_id, "resolution": 720},
        "result": result,
        "elapsed_ms": round(elapsed_ms, 3),
        "screenshot": screenshot,
        "resolution": resolution,
        "sha256": sha256,
        "captured_at": captured_at.isoformat(),
        "screen_context": screen_context,
        "labels": [],
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    device_name: str,
    resolution: str,
    screen_context: str,
    interval_ms: int,
) -> dict[str, Any]:
    """把逐帧记录汇总成 annotation 文件的元信息段。"""

    ok = [item for item in records if item["result"] == "ok"]
    elapsed = [float(item["elapsed_ms"]) for item in ok]
    failures = len(records) - len(ok)
    return {
        "artifact": "annotation",
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "read_only": True,
        "device": device_name,
        "resolution": resolution,
        "screen_context": screen_context,
        "interval_ms": interval_ms,
        "frames": len(records),
        "failures": failures,
        "elapsed_ms_p95": _p95(elapsed) if elapsed else None,
        "elapsed_ms_mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
        "note": "只读采集：仅调用 screencap，未发送任何点击/按键；labels 留给标注环节填充。",
    }


def _p95(samples: Sequence[float]) -> float:
    """最近秩法 p95，与 scripts/verify.py 同定义。"""

    ordered = sorted(samples)
    rank = min(len(ordered), max(1, int(-(-0.95 * len(ordered) // 1))))
    return round(float(ordered[rank - 1]), 3)


def write_annotation(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    *,
    out_dir: Path,
    name: str,
) -> tuple[Path, Path, Path]:
    """写出 annotation json / jsonl 与 frames.sha256，返回三个路径。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    jsonl_path = out_dir / f"{name}.jsonl"
    sha_path = out_dir / "frames.sha256"

    payload = {**meta, "records": list(records)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    sha_path.write_text(
        "".join(f"{item['sha256']}  {Path(item['screenshot']).name}\n" for item in records),
        encoding="utf-8",
    )
    return json_path, jsonl_path, sha_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 2 只读帧采集器（不点击）")
    parser.add_argument("--frames", type=int, default=100, help="采集帧数")
    parser.add_argument(
        "--interval-ms", type=int, default=DEFAULT_INTERVAL_MS, help="相邻两帧的间隔毫秒"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/phase2"), help="输出目录")
    parser.add_argument("--name", default="annotation", help="annotation 文件名（不含扩展名）")
    parser.add_argument(
        "--screen-context",
        choices=SCREEN_CONTEXTS,
        default="unknown",
        help="采集时人工确认的画面类型，写进元信息供后续筛选",
    )
    return parser.parse_args(argv)


async def collect(
    *,
    frames: int,
    interval_ms: int,
    out_dir: Path,
    screen_context: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """连设备抓帧；任何一次失败都记账，连续失败超上限就提前停。"""

    settings = load_settings()
    records: list[dict[str, Any]] = []
    frames_dir = out_dir / "frames"
    consecutive_failures = 0
    resolution = "unknown"
    device_name = "unknown"

    async with McpStdioClient(settings.mcp) as client:
        device = await MaaMcpDevice.connect(client, settings.device)
        device_name = device.name
        for index in range(1, frames + 1):
            started = time.perf_counter()
            try:
                frame: Frame = await device.screenshot()
            except Exception as exc:  # 采集中断也要落盘已有记录
                consecutive_failures += 1
                records.append(
                    frame_record(
                        step=index,
                        screenshot="",
                        resolution=resolution,
                        sha256="",
                        captured_at=datetime.now(LOCAL_TZ),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        controller_id=device.controller_id,
                        screen_context=screen_context,
                        result=f"fail:{type(exc).__name__}",
                    )
                )
                print(f"[{index}/{frames}] 截图失败：{exc}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"连续 {consecutive_failures} 次失败，提前停止")
                    break
                continue
            consecutive_failures = 0
            width, height = size_of(frame)
            resolution = f"{width}x{height}"
            path = frames_dir / f"frame_{index:04d}.png"
            write_bgr(path, frame)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(
                frame_record(
                    step=index,
                    screenshot=str(path).replace("\\", "/"),
                    resolution=resolution,
                    sha256=digest,
                    captured_at=datetime.now(LOCAL_TZ),
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    controller_id=device.controller_id,
                    screen_context=screen_context,
                )
            )
            if index % 10 == 0 or index == frames:
                print(f"[{index}/{frames}] {resolution} 已落盘 {path.name}")
            if interval_ms > 0 and index < frames:
                await asyncio.sleep(interval_ms / 1000)

    meta = summarize(
        records,
        device_name=device_name,
        resolution=resolution,
        screen_context=screen_context,
        interval_ms=interval_ms,
    )
    return records, meta


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records, meta = asyncio.run(
        collect(
            frames=args.frames,
            interval_ms=args.interval_ms,
            out_dir=args.out_dir,
            screen_context=args.screen_context,
        )
    )
    json_path, jsonl_path, sha_path = write_annotation(
        records, meta, out_dir=args.out_dir, name=args.name
    )
    print(
        f"采集完成：{meta['frames']} 帧，失败 {meta['failures']}，"
        f"p95 {meta['elapsed_ms_p95']}ms，分辨率 {meta['resolution']}"
    )
    print(f"已写出：{json_path}")
    print(f"        {jsonl_path}")
    print(f"        {sha_path}")
    return 1 if meta["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""命令行入口：doctor / capture / tap。"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Annotated

import typer

from arknights_agent.config import load_settings
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import write_bgr
from arknights_agent.mcp.client import McpStdioClient

app = typer.Typer(add_completion=False, help="明日方舟自我进化 Agent 的命令行入口")


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="输出机器可读 JSON")] = False,
    frames: Annotated[int, typer.Option(min=1, help="测截图延迟的帧数")] = 100,
) -> None:
    """检查 MCP、MAA、设备连接与截图延迟，输出 JSON 诊断结果。"""

    del json_output  # doctor 的 stdout 契约恒为 JSON，--json 仅为显式声明
    report = asyncio.run(_doctor(frames))
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    healthy = report["mcp"] == report["maa"] == report["device"] == "ok"
    raise typer.Exit(code=0 if healthy else 1)


@app.command()
def capture(out: Annotated[Path, typer.Option(help="保存路径")] = Path("runs/capture.png")) -> None:
    """连接设备并保存一张截图。"""

    async def run() -> Path:
        settings = load_settings()
        async with McpStdioClient(settings.mcp) as client:
            device = await MaaMcpDevice.connect(client, settings.device)
            frame = await device.screenshot()
            write_bgr(out, frame)
        return out

    saved = asyncio.run(run())
    typer.echo(f"已保存：{saved.resolve()}")


@app.command()
def tap(
    x: Annotated[int, typer.Option(help="逻辑坐标 X")],
    y: Annotated[int, typer.Option(help="逻辑坐标 Y")],
) -> None:
    """在逻辑坐标 (x, y) 处点击一次。"""

    async def run() -> bool:
        settings = load_settings()
        async with McpStdioClient(settings.mcp) as client:
            device = await MaaMcpDevice.connect(client, settings.device)
            return await device.tap(x, y)

    ok = asyncio.run(run())
    typer.echo(f"点击 ({x}, {y}) -> {'成功' if ok else '失败'}")
    raise typer.Exit(code=0 if ok else 1)


async def _doctor(frames: int) -> dict[str, object]:
    settings = load_settings()
    report: dict[str, object] = {"mcp": "fail", "maa": "fail", "device": "fail"}
    async with McpStdioClient(settings.mcp) as client:
        report["mcp"] = "ok"
        report["tools"] = len(await client.list_tools())
        device = await MaaMcpDevice.connect(client, settings.device)
        report["maa"] = "ok"
        report["device"] = "ok"
        report["device_name"] = device.name
        report["controller_id"] = device.controller_id
        report["native_size"] = list(device.native_size)
        # 逐帧截图取延迟样本；最近秩法 p95 的定义与 scripts/verify.py 一致。
        shots: list[float] = []
        for _ in range(frames):
            started = time.perf_counter()
            frame = await device.screenshot()
            shots.append((time.perf_counter() - started) * 1000.0)
        report["frames"] = len(shots)
        report["screenshot_p95_ms"] = round(sorted(shots)[math.ceil(0.95 * len(shots)) - 1], 3)
        height, width = frame.shape[:2]
        report["resolution"] = f"{width}x{height}"
    return report


def main() -> None:
    """`arknights-agent` 控制台脚本入口。"""

    app()

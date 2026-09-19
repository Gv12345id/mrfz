"""阶段 1 验收脚本：Python 通过 MCP 调用 MAA。

做三件事：
1. 启动 MCP server（MaaMCP）并连接模拟器；
2. 取一张截图，打印其尺寸并保存到 runs/；
3. 在指定逻辑坐标点击一次，并用点击前后的画面差异确认点击生效。

用法::

    python test_mcp_connection.py                 # 只验证连接与截图
    python test_mcp_connection.py --click 847 526 # 截图后在 (847, 526) 点击一次
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from arknights_agent.config import load_settings
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import diff_ratio, size_of, write_bgr
from arknights_agent.mcp.client import McpStdioClient

RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class CliArgs:
    """命令行参数。"""

    click: tuple[int, int] | None
    save: bool


def parse_args(argv: list[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(description="验证 Python -> MCP -> MAA -> 模拟器 链路")
    parser.add_argument(
        "--click",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        default=None,
        help="点击的逻辑坐标（1280x720 坐标系）；不传则只截图",
    )
    parser.add_argument("--no-save", action="store_true", help="不保存截图到 runs/")
    namespace = parser.parse_args(argv)
    raw_click: object = namespace.click
    click: tuple[int, int] | None = None
    if isinstance(raw_click, list) and len(raw_click) == 2:
        click = (int(raw_click[0]), int(raw_click[1]))
    return CliArgs(click=click, save=not bool(namespace.no_save))


async def run(click: tuple[int, int] | None, save: bool) -> int:
    settings = load_settings()
    runs_dir = RUNS_DIR

    async with McpStdioClient(settings.mcp) as client:
        tool_names = await client.list_tools()
        print(f"[1/4] MCP server 已就绪，暴露 {len(tool_names)} 个工具")

        device = await MaaMcpDevice.connect(client, settings.device)
        print(
            f"[2/4] 已连接设备：{device.name}"
            f"（controller_id={device.controller_id}，"
            f"原生分辨率 {device.native_size[0]}x{device.native_size[1]}）"
        )

        before = await device.screenshot()
        width, height = size_of(before)
        print(
            f"[3/4] 截图成功：shape={before.shape} dtype={before.dtype} "
            f"尺寸={width}x{height}（逻辑坐标系 {device.logical_size[0]}x{device.logical_size[1]}）"
        )
        if save:
            write_bgr(runs_dir / "screenshot.png", before)
            print(f"      已保存：{(runs_dir / 'screenshot.png').resolve()}")

        if click is None:
            print("[4/4] 未指定 --click，跳过点击")
            return 0

        x, y = click
        native = device.to_native(x, y)
        ok = await device.tap(x, y)
        print(f"[4/4] 点击逻辑坐标 ({x}, {y}) -> 原生坐标 {native} -> {'成功' if ok else '失败'}")
        if not ok:
            return 1

        await asyncio.sleep(1.0)
        after = await device.screenshot()
        ratio = diff_ratio(before, after)
        print(f"      点击后画面变化像素占比：{ratio:.2%}（>0 说明点击已被游戏响应）")
        if save:
            write_bgr(runs_dir / "screenshot_after_click.png", after)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args.click, save=args.save))


if __name__ == "__main__":
    sys.exit(main())

"""离线 MaaMCP 替身：实现本阶段用到的四个工具，供离线测试驱动完整 MCP 链路。

行为对齐真实 MaaMCP：
- ``find_adb_device_list`` 返回设备名列表；
- ``connect_adb_device`` 返回 controller_id；
- ``screencap`` 把 PNG 写到 ``FAKE_MCP_DIR``，返回文件路径；``resolution=720``
  时按短边归一化（1920x1080 -> 1280x720），``resolution=None`` 时返回原生图；
- ``click`` 记录点击到 ``FAKE_MCP_DIR/clicks.jsonl`` 并返回 True。

点击之后生成的截图会画上标记，便于验证"点击后画面确实变化"。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

FAKE_MCP_DIR_ENV = "FAKE_MCP_DIR"
NATIVE_SIZE = (1920, 1080)
DEVICE_NAME = "FakeEmulator12-127.0.0.1:16416"

mcp = FastMCP("FakeMaaMCP")


def _work_dir() -> Path:
    raw = os.environ.get(FAKE_MCP_DIR_ENV)
    if not raw:
        raise RuntimeError(f"环境变量 {FAKE_MCP_DIR_ENV} 未设置")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clicks() -> list[dict[str, int]]:
    log = _work_dir() / "clicks.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def _render() -> np.ndarray:
    """生成一帧可区分的图像；有点击记录时在对应位置画标记。"""

    width, height = NATIVE_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (32, 32, 64)
    cv2.rectangle(frame, (40, 40), (300, 200), (200, 200, 200), -1)
    for record in _clicks():
        x, y = record["x"], record["y"]
        cv2.circle(frame, (x, y), 40, (0, 0, 255), -1)
    return frame


@mcp.tool(name="find_adb_device_list")
def find_adb_device_list() -> list[str]:
    """返回唯一的假设备。"""

    return [DEVICE_NAME]


@mcp.tool(name="connect_adb_device")
def connect_adb_device(device_name: str) -> str | None:
    """返回固定的 controller_id；设备名不匹配时返回 None。"""

    return "ctrl_fake" if device_name == DEVICE_NAME else None


@mcp.tool(name="screencap")
def screencap(
    controller_id: str,
    region: tuple[int, int, int, int] | None = None,
    resolution: int | None = 720,
) -> str | None:
    """落盘一张截图并返回路径。"""

    if controller_id != "ctrl_fake":
        return None
    frame = _render()
    if region is not None:
        x, y, w, h = region
        frame = frame[y : y + h, x : x + w]
    if resolution is not None:
        height, width = frame.shape[:2]
        scale = resolution / min(height, width)
        frame = cv2.resize(
            frame,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    index = len(list(_work_dir().glob("shot_*.png")))
    path = _work_dir() / f"shot_{index:04d}.png"
    cv2.imwrite(str(path), frame)
    return str(path)


@mcp.tool(name="click")
def click(controller_id: str, x: int, y: int, button: int = 0, duration: int = 50) -> bool:
    """记录点击并返回 True。"""

    if controller_id != "ctrl_fake":
        return False
    log = _work_dir() / "clicks.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"x": x, "y": y, "button": button, "duration": duration}) + "\n")
    return True


if __name__ == "__main__":
    mcp.run()

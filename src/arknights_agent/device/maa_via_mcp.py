"""基于 MaaMCP 的设备后端实现。

坐标约定：
- 对外（逻辑坐标系）统一 1280x720，与项目其余部分一致。
- MaaMCP 的 ``click`` 需要设备原生分辨率坐标，因此内部用 ``to_native`` 换算。
- ``screencap`` 的 ``resolution`` 参数按短边归一化；短边 720 时 16:9 设备
  正好得到 1280x720。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arknights_agent.config import DeviceConfig
from arknights_agent.imaging import Frame, read_bgr, size_of
from arknights_agent.mcp.client import McpStdioClient, McpToolError
from arknights_agent.mcp.tools import (
    CLICK,
    CONNECT_ADB_DEVICE,
    FIND_ADB_DEVICE_LIST,
    SCREENCAP,
    SWIPE,
)

# screencap 的短边归一化目标：720 像素
SHORT_EDGE_720 = 720


class DeviceError(RuntimeError):
    """设备发现、连接、截图或点击失败。"""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DeviceError(f"设备列表格式非预期：{value!r}")
    return [str(item) for item in value]


def select_device(names: Sequence[str], match: str | None) -> str:
    """按配置片段选择设备。

    ``match`` 为空时只在恰好一台设备的情况下自动选中；发现多台则报错，
    避免"随手选一台"把操作打到错误设备上。
    """

    if match:
        hits = [name for name in names if match in name]
        if not hits:
            raise DeviceError(f"没有匹配 {match!r} 的设备；可用设备：{list(names)}")
        if len(hits) > 1:
            raise DeviceError(f"匹配 {match!r} 的设备不止一台：{hits}，请用更精确的片段")
        return hits[0]

    if len(names) == 1:
        return names[0]
    if not names:
        raise DeviceError("未发现任何 ADB 设备，请确认模拟器已启动且 ADB 可连接")
    raise DeviceError(f"发现多台设备：{list(names)}，请在配置中指定 device.match")


class MaaMcpDevice:
    """通过 MCP 调用 MaaMCP 完成截图与点击的 :class:`DeviceBackend` 实现。"""

    def __init__(
        self,
        *,
        client: McpStdioClient,
        name: str,
        controller_id: str,
        native_size: tuple[int, int],
        logical_size: tuple[int, int],
        click_duration_ms: int = 50,
    ) -> None:
        self._client = client
        self._name = name
        self._controller_id = controller_id
        self._native_size = native_size
        self._logical_size = logical_size
        self._click_duration_ms = click_duration_ms
        self._closed = False

    @classmethod
    async def connect(cls, client: McpStdioClient, config: DeviceConfig) -> MaaMcpDevice:
        """扫描设备、建立控制器，并测量原生分辨率。"""

        names = _as_str_list(await client.call_value(FIND_ADB_DEVICE_LIST))
        selected = select_device(names, config.match)

        controller_id = await client.call_value(CONNECT_ADB_DEVICE, {"device_name": selected})
        if not isinstance(controller_id, str) or not controller_id:
            raise DeviceError(f"连接设备 {selected!r} 失败（未返回 controller_id）")

        native_size = await _measure_native_size(client, controller_id)
        return cls(
            client=client,
            name=selected,
            controller_id=controller_id,
            native_size=native_size,
            logical_size=(config.logical_width, config.logical_height),
            click_duration_ms=config.click_duration_ms,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def controller_id(self) -> str:
        return self._controller_id

    @property
    def native_size(self) -> tuple[int, int]:
        return self._native_size

    @property
    def logical_size(self) -> tuple[int, int]:
        return self._logical_size

    def to_native(self, x: int, y: int) -> tuple[int, int]:
        """逻辑坐标 -> 原生坐标，并钳制到屏幕范围内。"""

        logical_w, logical_h = self._logical_size
        native_w, native_h = self._native_size
        native_x = round(x * native_w / logical_w)
        native_y = round(y * native_h / logical_h)
        return (
            max(0, min(native_x, native_w - 1)),
            max(0, min(native_y, native_h - 1)),
        )

    def _ensure_open(self) -> None:
        """关闭后继续调用属于编程错误，直接报错而不是静默打到别的会话。"""

        if self._closed:
            raise DeviceError(f"设备 {self._name!r} 已关闭，请重新 connect()")

    async def screenshot(self) -> Frame:
        """截图并返回逻辑坐标系下的 BGR 数组。"""

        self._ensure_open()
        path = await self._client.call_text(
            SCREENCAP,
            {"controller_id": self._controller_id, "resolution": SHORT_EDGE_720},
        )
        frame = read_bgr(_as_path(path))
        width, height = size_of(frame)
        if (width, height) != self._logical_size:
            raise DeviceError(
                f"截图为 {width}x{height}，与逻辑坐标系 "
                f"{self._logical_size[0]}x{self._logical_size[1]} 不一致；"
                "请确认设备为 16:9 且游戏处于横屏"
            )
        return frame

    async def tap(self, x: int, y: int) -> bool:
        """在逻辑坐标 ``(x, y)`` 处点击一次。"""

        self._ensure_open()
        native_x, native_y = self.to_native(x, y)
        return await self._client.call_bool(
            CLICK,
            {
                "controller_id": self._controller_id,
                "x": native_x,
                "y": native_y,
                "duration": self._click_duration_ms,
            },
        )

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        """在逻辑坐标之间滑动；MaaMCP 只认原生坐标，四个点都要换算。"""

        self._ensure_open()
        native_x1, native_y1 = self.to_native(x1, y1)
        native_x2, native_y2 = self.to_native(x2, y2)
        return await self._client.call_bool(
            SWIPE,
            {
                "controller_id": self._controller_id,
                "start_x": native_x1,
                "start_y": native_y1,
                "end_x": native_x2,
                "end_y": native_y2,
                "duration": duration_ms,
            },
        )

    def close(self) -> None:
        """幂等关闭本地设备句柄。

        MaaMCP 没有"断开设备"工具（实测 24 个工具里只有 ``connect_adb_device``），
        因此这里只做本地状态收口；真正的资源释放由 MaaMCP 进程退出承担。
        """

        self._closed = True


def _as_path(value: str) -> Path:
    """把 MaaMCP 返回的截图路径转成 :class:`Path`，不存在时明确报错。"""

    path = Path(value)
    if path.suffix.lower() != ".png":
        raise DeviceError(f"screencap 返回的不是 PNG 路径：{value!r}")
    if not path.is_file():
        raise DeviceError(f"screencap 返回的截图文件不存在：{path}")
    return path


async def _measure_native_size(client: McpStdioClient, controller_id: str) -> tuple[int, int]:
    """用一次不做归一化的截图（resolution=None）测量设备原生分辨率。"""

    try:
        path = await client.call_text(
            SCREENCAP, {"controller_id": controller_id, "resolution": None}
        )
    except McpToolError as exc:  # pragma: no cover - 依赖真实 server 行为
        raise DeviceError(f"测量原生分辨率失败：{exc}") from exc
    frame = read_bgr(_as_path(path))
    return size_of(frame)

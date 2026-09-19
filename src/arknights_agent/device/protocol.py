"""设备后端接口。

任务层只依赖这个 Protocol，因此替换后端（MaaMCP / 离线回放 / 其它）无需改动上层。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from arknights_agent.imaging import Frame


@runtime_checkable
class DeviceBackend(Protocol):
    """截图与点击的最小接口。"""

    @property
    def name(self) -> str:
        """设备名称（来自 ADB 扫描结果）。"""
        ...

    @property
    def native_size(self) -> tuple[int, int]:
        """设备原生分辨率 ``(width, height)``。"""
        ...

    @property
    def logical_size(self) -> tuple[int, int]:
        """逻辑坐标系尺寸 ``(width, height)``，默认 1280x720。"""
        ...

    async def screenshot(self) -> Frame:
        """截取当前屏幕，返回逻辑坐标系下的 BGR 数组。"""
        ...

    async def tap(self, x: int, y: int) -> bool:
        """在逻辑坐标系 ``(x, y)`` 处点击一次，返回是否成功下发。"""
        ...

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        """在逻辑坐标 ``(x1, y1)`` 到 ``(x2, y2)`` 之间滑动，返回是否成功下发。"""
        ...

    def close(self) -> None:
        """关闭后端并释放本地资源；必须幂等，关闭后再调用其它方法应报错。"""
        ...

    def to_native(self, x: int, y: int) -> tuple[int, int]:
        """把逻辑坐标换算为设备原生坐标。"""
        ...

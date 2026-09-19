"""离线设备后端：按顺序回放帧序列，并记录点击。

所有不依赖真机的测试都走这里，保证 `pytest -m "not integration"` 能在任何机器上跑。
"""

from __future__ import annotations

from collections.abc import Sequence

from arknights_agent.imaging import Frame


class FakeDevice:
    """回放固定帧序列的 :class:`~arknights_agent.device.protocol.DeviceBackend`。"""

    def __init__(
        self,
        frames: Sequence[Frame],
        *,
        name: str = "FakeDevice",
        native_size: tuple[int, int] = (1920, 1080),
        logical_size: tuple[int, int] = (1280, 720),
        tap_results: Sequence[bool] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("frames 不能为空")
        self._frames = list(frames)
        self._name = name
        self._native_size = native_size
        self._logical_size = logical_size
        self._tap_results = list(tap_results) if tap_results is not None else []
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []
        self.closed = False
        self._cursor = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_size(self) -> tuple[int, int]:
        return self._native_size

    @property
    def logical_size(self) -> tuple[int, int]:
        return self._logical_size

    def to_native(self, x: int, y: int) -> tuple[int, int]:
        logical_w, logical_h = self._logical_size
        native_w, native_h = self._native_size
        return (
            max(0, min(round(x * native_w / logical_w), native_w - 1)),
            max(0, min(round(y * native_h / logical_h), native_h - 1)),
        )

    async def screenshot(self) -> Frame:
        """返回下一帧；帧用尽后一直返回最后一帧。"""

        if self.closed:
            raise RuntimeError("FakeDevice 已关闭")
        frame = self._frames[min(self._cursor, len(self._frames) - 1)]
        self._cursor += 1
        return frame

    async def tap(self, x: int, y: int) -> bool:
        if self.closed:
            raise RuntimeError("FakeDevice 已关闭")
        index = len(self.taps)
        self.taps.append((x, y))
        if index < len(self._tap_results):
            return self._tap_results[index]
        return True

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
        """记录一次逻辑坐标滑动；关闭后抛错。"""

        if self.closed:
            raise RuntimeError("FakeDevice 已关闭")
        self.swipes.append((x1, y1, x2, y2, duration_ms))
        return True

    def close(self) -> None:
        """幂等关闭。"""

        self.closed = True

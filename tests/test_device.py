"""设备层测试：设备选择、坐标换算、截图尺寸校验。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from arknights_agent.device.fake import FakeDevice
from arknights_agent.device.maa_via_mcp import (
    DeviceError,
    MaaMcpDevice,
    _as_path,
    select_device,
)
from arknights_agent.imaging import write_bgr
from arknights_agent.mcp.tools import SWIPE


def test_select_device_by_match() -> None:
    names = ["MuMuEmulator12-127.0.0.1:16416", "MuMuEmulator12-127.0.0.1:7555"]
    assert select_device(names, "16416") == names[0]


def test_select_device_single_auto() -> None:
    assert select_device(["only-one"], None) == "only-one"


def test_select_device_multiple_requires_match() -> None:
    with pytest.raises(DeviceError, match="发现多台设备"):
        select_device(["a", "b"], None)


def test_select_device_no_match_message_lists_devices() -> None:
    with pytest.raises(DeviceError, match="没有匹配"):
        select_device(["a", "b"], "zzz")


def test_select_device_ambiguous_match_rejected() -> None:
    with pytest.raises(DeviceError, match="不止一台"):
        select_device(["MuMu-1", "MuMu-2"], "MuMu")


def test_select_device_empty_list() -> None:
    with pytest.raises(DeviceError, match="未发现任何 ADB 设备"):
        select_device([], None)


def test_to_native_scales_and_clamps() -> None:
    device = _device()
    assert device.to_native(0, 0) == (0, 0)
    assert device.to_native(640, 360) == (960, 540)
    assert device.to_native(1280, 720) == (1919, 1079)
    assert device.to_native(-10, -10) == (0, 0)
    assert device.to_native(99999, 99999) == (1919, 1079)


def test_fake_device_records_taps() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake = FakeDevice([frame], tap_results=[False, True])
    assert asyncio.run(fake.tap(1, 2)) is False
    assert asyncio.run(fake.tap(3, 4)) is True
    assert fake.taps == [(1, 2), (3, 4)]


def test_swipe_converts_all_four_coordinates_to_native() -> None:
    client = _RecordingClient()
    device = _device(client)

    ok = asyncio.run(device.swipe(10, 20, 640, 360, 300))

    assert ok is True
    assert client.calls == [
        (
            SWIPE,
            {
                "controller_id": "ctrl",
                "start_x": 15,
                "start_y": 30,
                "end_x": 960,
                "end_y": 540,
                "duration": 300,
            },
        )
    ]


def test_close_is_idempotent_and_blocks_later_calls() -> None:
    device = _device(_RecordingClient())

    device.close()
    device.close()

    with pytest.raises(DeviceError, match="已关闭"):
        asyncio.run(device.tap(1, 1))
    with pytest.raises(DeviceError, match="已关闭"):
        asyncio.run(device.swipe(1, 1, 2, 2, 100))


def test_fake_device_records_swipes_and_closes() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake = FakeDevice([frame])

    assert asyncio.run(fake.swipe(1, 2, 3, 4, 250)) is True
    fake.close()
    fake.close()

    assert fake.swipes == [(1, 2, 3, 4, 250)]
    with pytest.raises(RuntimeError, match="已关闭"):
        asyncio.run(fake.tap(1, 1))


def test_screenshot_rejects_non_logical_size(tmp_path: Path) -> None:
    portrait = np.zeros((1280, 720, 3), dtype=np.uint8)
    path = tmp_path / "portrait.png"
    write_bgr(path, portrait)
    device = MaaMcpDevice(
        client=_FixedPathClient(path),  # type: ignore[arg-type]
        name="fake",
        controller_id="ctrl",
        native_size=(1080, 1920),
        logical_size=(1280, 720),
    )
    with pytest.raises(DeviceError, match="与逻辑坐标系"):
        asyncio.run(device.screenshot())


def test_as_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DeviceError, match="不存在"):
        _as_path(str(tmp_path / "nope.png"))


def _device(client: object | None = None) -> MaaMcpDevice:
    return MaaMcpDevice(
        client=client if client is not None else _NeverCalledClient(),  # type: ignore[arg-type]
        name="fake",
        controller_id="ctrl",
        native_size=(1920, 1080),
        logical_size=(1280, 720),
    )


class _NeverCalledClient:
    """仅供 to_native 等纯坐标测试使用的占位客户端。"""

    async def call_text(self, name: str, arguments: object = None) -> str:
        raise AssertionError("不应被调用")

    async def call_bool(self, name: str, arguments: object = None) -> bool:
        raise AssertionError("不应被调用")


class _RecordingClient:
    """记录 call_bool 的 (工具名, 参数)，用于断言坐标换算后的实际请求。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_bool(self, name: str, arguments: dict[str, object] | None = None) -> bool:
        self.calls.append((name, dict(arguments or {})))
        return True


class _FixedPathClient:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def call_text(self, name: str, arguments: object = None) -> str:
        return str(self._path)

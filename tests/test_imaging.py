"""图像工具测试。"""

from __future__ import annotations

import numpy as np

from arknights_agent.imaging import diff_ratio, read_bgr, size_of, write_bgr


def test_size_of() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert size_of(frame) == (1280, 720)


def test_diff_ratio_identical_is_zero() -> None:
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    assert diff_ratio(frame, frame.copy()) == 0.0


def test_diff_ratio_ignores_subtle_noise() -> None:
    before = np.full((720, 1280, 3), 128, dtype=np.uint8)
    after = before.copy()
    after[:100, :100] = 132  # 灰度差 4，低于默认阈值 12
    assert diff_ratio(before, after) == 0.0


def test_diff_ratio_detects_change() -> None:
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[:50, :] = 255  # 一半像素发生大变化
    assert diff_ratio(before, after) == 0.5


def test_diff_ratio_shape_mismatch_is_full() -> None:
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = np.zeros((200, 200, 3), dtype=np.uint8)
    assert diff_ratio(before, after) == 1.0


def test_write_then_read_roundtrip(tmp_path) -> None:
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    path = tmp_path / "x.png"
    write_bgr(path, frame)
    assert np.array_equal(read_bgr(path), frame)

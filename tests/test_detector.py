"""`perception.detector` 的纯逻辑测试：置信度过滤、越界裁剪、排序与几何属性。

`YoloDetector` 的 ultralytics 调用不在离线测试里跑（需要权重与 torch 推理），
这里只验证它的构造期错误处理；框的后处理全部走 `enemy_boxes()`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arknights_agent.perception.detector import (
    DetectorError,
    EnemyBox,
    YoloDetector,
    enemy_boxes,
)

FRAME_SIZE = (1280, 720)


def test_enemy_box_geometry() -> None:
    box = EnemyBox(x1=100.0, y1=200.0, x2=180.0, y2=280.0, confidence=0.9)

    assert (box.width, box.height) == (80.0, 80.0)
    assert box.center == (140.0, 240.0)
    assert box.area == 6400.0
    assert box.is_elite is False


def test_elite_class_marks_box_as_elite() -> None:
    assert EnemyBox(0, 0, 10, 10, 0.5, class_id=1).is_elite is True


def test_rows_below_min_confidence_are_dropped() -> None:
    rows = [
        [100.0, 100.0, 200.0, 200.0, 0.80, 0.0],
        [300.0, 300.0, 400.0, 400.0, 0.24, 0.0],
    ]

    boxes = enemy_boxes(rows, frame_size=FRAME_SIZE, min_confidence=0.25)

    assert len(boxes) == 1
    assert boxes[0].confidence == pytest.approx(0.80)


def test_row_exactly_at_threshold_is_kept() -> None:
    rows = [[100.0, 100.0, 200.0, 200.0, 0.25, 0.0]]

    assert len(enemy_boxes(rows, frame_size=FRAME_SIZE, min_confidence=0.25)) == 1


def test_boxes_are_clipped_into_frame() -> None:
    rows = [[-50.0, -20.0, 200.0, 200.0, 0.9, 0.0]]

    box = enemy_boxes(rows, frame_size=FRAME_SIZE, min_confidence=0.25)[0]

    assert (box.x1, box.y1, box.x2, box.y2) == (0.0, 0.0, 200.0, 200.0)


def test_boxes_outside_frame_are_dropped() -> None:
    rows = [
        [2000.0, 900.0, 2100.0, 1000.0, 0.9, 0.0],
        [-400.0, -300.0, -200.0, -100.0, 0.9, 0.0],
    ]

    assert enemy_boxes(rows, frame_size=FRAME_SIZE, min_confidence=0.25) == []


def test_boxes_are_sorted_by_area_descending() -> None:
    rows = [
        [100.0, 100.0, 140.0, 140.0, 0.9, 0.0],
        [300.0, 300.0, 500.0, 500.0, 0.5, 0.0],
        [600.0, 600.0, 700.0, 620.0, 0.5, 0.0],
    ]

    areas = [box.area for box in enemy_boxes(rows, frame_size=FRAME_SIZE, min_confidence=0.25)]

    assert areas == sorted(areas, reverse=True)
    assert areas[0] == 40000.0


def test_empty_rows_yield_no_boxes() -> None:
    assert enemy_boxes([], frame_size=FRAME_SIZE, min_confidence=0.25) == []


def test_missing_weights_raise_detector_error(tmp_path: Path) -> None:
    with pytest.raises(DetectorError, match="权重不存在"):
        YoloDetector(tmp_path / "nope.pt")

"""`perception.pipeline.Perception` 的离线测试：分辨率校验、限频缓存、框归一化。"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arknights_agent.perception.detector import EnemyBox
from arknights_agent.perception.pipeline import Perception, PerceptionError
from arknights_agent.perception.state import MAX_OPERATORS, STATE_DIM

LOGICAL_SIZE = (1280, 720)


def blank_frame(width: int = 1280, height: int = 720) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeDetector:
    """可控检测器：记录调用次数，返回预置框。"""

    def __init__(self, boxes: list[EnemyBox] | None = None) -> None:
        self.boxes = boxes or []
        self.calls = 0

    def detect(self, frame: NDArray[np.uint8]) -> list[EnemyBox]:
        self.calls += 1
        return list(self.boxes)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_read_returns_state_with_expected_dimension() -> None:
    perception = Perception(FakeDetector())

    vector = perception.read(blank_frame()).to_vector()

    assert len(vector) == STATE_DIM
    assert vector == [0.0] * STATE_DIM


def test_frame_size_mismatch_is_rejected() -> None:
    """ROI 按 1280×720 写死，别的分辨率必须报错而不是悄悄缩放。"""

    perception = Perception(FakeDetector())

    with pytest.raises(PerceptionError, match="不一致"):
        perception.read(blank_frame(1920, 1080))


def test_enemy_box_is_normalised_to_unit_range() -> None:
    detector = FakeDetector([EnemyBox(320.0, 180.0, 640.0, 360.0, 0.9)])
    perception = Perception(detector)

    slot = perception.read(blank_frame()).enemies[0]

    assert (slot.x, slot.y) == pytest.approx((0.375, 0.375))
    assert (slot.w, slot.h) == pytest.approx((0.25, 0.25))
    assert slot.alive is True


def test_out_of_frame_box_is_clipped_before_normalising() -> None:
    detector = FakeDetector([EnemyBox(-100.0, -100.0, 128.0, 72.0, 0.9)])
    perception = Perception(detector)

    slot = perception.read(blank_frame()).enemies[0]

    assert (slot.x, slot.y) == pytest.approx((0.05, 0.05))
    assert 0.0 <= slot.x <= 1.0 and 0.0 <= slot.w <= 1.0


def test_operators_are_padded_to_twelve_slots() -> None:
    perception = Perception(FakeDetector())

    state = perception.read(blank_frame())

    assert len(state.operators) == MAX_OPERATORS
    assert all(slot.deployed is False for slot in state.operators)


def test_cost_is_unknown_without_templates() -> None:
    perception = Perception(FakeDetector())

    state = perception.read(blank_frame())

    assert state.cost is None
    assert state.to_vector()[120] == 0.0


def test_repeated_reads_within_interval_reuse_cache() -> None:
    clock = FakeClock()
    detector = FakeDetector([EnemyBox(100.0, 100.0, 200.0, 200.0, 0.9)])
    perception = Perception(detector, min_interval_s=0.5, clock=clock)

    first = perception.read(blank_frame())
    clock.now = 0.2
    second = perception.read(blank_frame())

    assert detector.calls == 1
    assert second is first


def test_read_after_interval_runs_detector_again() -> None:
    clock = FakeClock()
    detector = FakeDetector([EnemyBox(100.0, 100.0, 200.0, 200.0, 0.9)])
    perception = Perception(detector, min_interval_s=0.5, clock=clock)

    perception.read(blank_frame())
    clock.now = 0.6
    perception.read(blank_frame())

    assert detector.calls == 2


def test_interval_zero_disables_caching() -> None:
    detector = FakeDetector()
    perception = Perception(detector, min_interval_s=0.0)

    perception.read(blank_frame())
    perception.read(blank_frame())

    assert detector.calls == 2

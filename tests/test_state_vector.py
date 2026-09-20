"""`BattleState.to_vector()` 的维度、补零/截断与值域测试。

对应 PLANS.md 阶段 2 验收 3：维度 = 173；补零与截断行为符合定义表；全部值域在 [0, 1]；
1000 帧回放 100% 通过 pydantic 校验。维度布局是破坏性接口，这里逐段锁住偏移量。
"""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from arknights_agent.perception.state import (
    COST_DIVISOR,
    ENEMY_REMAINING_DIVISOR,
    LEAK_DIVISOR,
    MAX_ENEMIES,
    MAX_OPERATORS,
    STATE_DIM,
    BattleState,
    EnemySlot,
    GlobalState,
    OperatorSlot,
)

ENEMY_SECTION = MAX_ENEMIES * 6
COST_INDEX = ENEMY_SECTION
OPERATOR_START = COST_INDEX + 1
GLOBAL_START = OPERATOR_START + MAX_OPERATORS * 4


def enemy_index(slot: int) -> int:
    return slot * 6


def operator_index(slot: int) -> int:
    return OPERATOR_START + slot * 4


def test_state_dim_is_173() -> None:
    assert STATE_DIM == 173
    assert len(BattleState().to_vector()) == STATE_DIM


def test_empty_state_is_all_zeros() -> None:
    """没有敌人、没有费用、没有干员时整条向量为 0（不能是 None 或 NaN）。"""

    assert BattleState().to_vector() == [0.0] * STATE_DIM


def test_default_state_vector_length_matches_section_offsets() -> None:
    """区段偏移写死在这里：改布局必须同时改 PLANS.md 与 state.py。"""

    assert ENEMY_SECTION == 120
    assert COST_INDEX == 120
    assert OPERATOR_START == 121
    assert GLOBAL_START == 169
    assert GLOBAL_START + 4 == STATE_DIM


def test_enemy_slot_is_encoded_in_declared_order() -> None:
    slot = EnemySlot(x=0.25, y=0.5, w=0.125, h=0.75, is_elite=True, alive=True, confidence=1.0)

    vector = BattleState(enemies=(slot,)).to_vector()

    assert vector[0:6] == [0.25, 0.5, 0.125, 0.75, 1.0, 1.0]


def test_dead_enemy_has_zero_alive_bit() -> None:
    slot = EnemySlot(x=0.1, y=0.1, w=0.1, h=0.1, alive=False)

    assert BattleState(enemies=(slot,)).to_vector()[5] == 0.0


def test_enemy_slots_beyond_twenty_are_dropped_by_confidence() -> None:
    """超过 20 个按置信度取前 20（PLANS.md：截断规则）。"""

    slots = tuple(
        EnemySlot(x=0.5, y=0.5, w=0.1, h=0.1, confidence=index / 100) for index in range(25)
    )

    vector = BattleState(enemies=slots).to_vector()

    # 最高的 5 个（confidence 0.24 … 0.20）应被保留在第 0…4 槽，最低的 5 个被丢掉。
    assert vector[enemy_index(0) + 5] == 1.0
    assert vector[enemy_index(19) + 5] == 1.0
    assert vector[enemy_index(20) :] == [0.0] * (STATE_DIM - ENEMY_SECTION)


def test_enemy_slots_are_padded_with_zeros() -> None:
    slots = tuple(
        EnemySlot(x=0.2, y=0.2, w=0.2, h=0.2, confidence=1.0 - index / 10) for index in range(3)
    )

    vector = BattleState(enemies=slots).to_vector()

    assert vector[enemy_index(3) : ENEMY_SECTION] == [0.0] * (ENEMY_SECTION - enemy_index(3))


def test_cost_is_normalised_by_99_and_clamped() -> None:
    assert BattleState(cost=0).to_vector()[COST_INDEX] == 0.0
    assert BattleState(cost=99).to_vector()[COST_INDEX] == 1.0
    assert BattleState(cost=50).to_vector()[COST_INDEX] == pytest.approx(50 / COST_DIVISOR)
    assert BattleState(cost=500).to_vector()[COST_INDEX] == 1.0


def test_unknown_cost_encodes_as_zero() -> None:
    """费用读不出时按下限编码——决策层看到「费用不够」，不会拿脏数据去点。"""

    assert BattleState(cost=None).to_vector()[COST_INDEX] == 0.0


def test_operator_slot_is_encoded_in_declared_order() -> None:
    slot = OperatorSlot(deployed=True, cooldown_remaining=5.0, cooldown_total=20.0, cost=14)

    vector = BattleState(operators=(slot,)).to_vector()

    assert vector[operator_index(0) : operator_index(0) + 4] == [
        1.0,
        0.25,
        14 / COST_DIVISOR,
        0.0,
    ]


def test_operator_cooldown_with_zero_total_is_not_nan() -> None:
    slot = OperatorSlot(deployed=True, cooldown_remaining=7.0, cooldown_total=0.0)

    assert BattleState(operators=(slot,)).to_vector()[operator_index(0) + 1] == 0.0


def test_operator_slots_beyond_twelve_are_truncated() -> None:
    slots = tuple(OperatorSlot(cost=index) for index in range(15))

    vector = BattleState(operators=slots).to_vector()

    assert vector[operator_index(11) + 2] == pytest.approx(11 / COST_DIVISOR)
    assert len(vector) == STATE_DIM


def test_global_section_is_encoded_in_declared_order() -> None:
    state = GlobalState(
        progress_s=30.0, total_s=120.0, kills=5, kill_target=10, leaks=1, enemies_remaining=10
    )

    vector = BattleState(global_state=state).to_vector()

    assert vector[GLOBAL_START:] == [
        0.25,
        0.5,
        1 / LEAK_DIVISOR,
        10 / ENEMY_REMAINING_DIVISOR,
    ]


def test_global_section_with_zero_denominators_is_not_nan() -> None:
    vector = BattleState(global_state=GlobalState()).to_vector()

    assert vector[GLOBAL_START:] == [0.0, 0.0, 0.0, 0.0]


def test_global_counters_are_clamped_to_unit_range() -> None:
    state = GlobalState(progress_s=999.0, total_s=60.0, kills=99, kill_target=1, leaks=9)

    vector = BattleState(global_state=state).to_vector()

    assert vector[GLOBAL_START] == 1.0
    assert vector[GLOBAL_START + 1] == 1.0
    assert vector[GLOBAL_START + 2] == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", -0.1),
        ("x", 1.1),
        ("w", 2.0),
        ("confidence", 1.5),
    ],
)
def test_enemy_slot_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        EnemySlot(**{"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5, field: value})


def test_negative_counters_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GlobalState(kills=-1)
    with pytest.raises(ValidationError):
        OperatorSlot(cost=-1)
    with pytest.raises(ValidationError):
        BattleState(cost=-1)


def test_thousand_frame_replay_stays_valid_and_in_range() -> None:
    """1000 帧回放：每帧都能通过 pydantic 校验，且向量值域全是 [0, 1]。"""

    generator = random.Random(20260920)
    for _ in range(1000):
        state = BattleState(
            enemies=tuple(
                EnemySlot(
                    x=generator.random(),
                    y=generator.random(),
                    w=generator.random(),
                    h=generator.random(),
                    is_elite=generator.random() < 0.2,
                    alive=generator.random() < 0.9,
                    confidence=generator.random(),
                )
                for _ in range(generator.randint(0, 26))
            ),
            cost=generator.choice([None, 0, 7, 42, 99, 120]),
            operators=tuple(
                OperatorSlot(
                    deployed=generator.random() < 0.5,
                    cooldown_remaining=generator.uniform(0, 70),
                    cooldown_total=generator.choice([0.0, 20.0, 70.0]),
                    cost=generator.randint(0, 40),
                    skill_ready=generator.random() < 0.5,
                )
                for _ in range(generator.randint(0, 14))
            ),
            global_state=GlobalState(
                progress_s=generator.uniform(0, 400),
                total_s=generator.choice([0.0, 180.0]),
                kills=generator.randint(0, 60),
                kill_target=generator.choice([0, 30]),
                leaks=generator.randint(0, 5),
                enemies_remaining=generator.randint(0, 40),
            ),
        )

        assert BattleState.model_validate(state.model_dump()) == state
        vector = state.to_vector()
        assert len(vector) == STATE_DIM
        assert all(0.0 <= value <= 1.0 for value in vector)

"""战场状态向量：`BattleState`（pydantic）+ `to_vector()`，维度固定 **173**。

维度布局来自 `PLANS.md` 阶段 2「状态向量定义」。**改动布局是破坏性变更**：
必须同时更新 PLANS.md 的表格、本文件的 `to_vector()` 与 `tests/test_state_vector.py`。

| 区段 | 维度 | 内容 |
| --- | --- | --- |
| 敌人 | 20 × 6 = 120 | 每槽 `x, y, w, h`（归一化 0-1）、类别（0 普通 / 1 精英）、存活位 |
| 费用 | 1 | 当前费用 / 99，钳制到 0-1 |
| 干员 | 12 × 4 = 48 | 部署位是否占用、CD 剩余 / 冷却上限、部署费用 / 99、技能是否就绪 |
| 全局 | 4 | 关卡进度 / 总时长、击杀数 / 目标、漏怪数 / 3、剩余敌人估计 / 20 |

两条编码约定（都是为了让决策层永远拿到 [0, 1] 的定长向量）：

1. **读不出即按下限**：费用读不出（`cost is None`）时该槽写 0——在动作掩码里表现为
   "费用不够、不能部署"，比拿一个可能错的费用去决策安全（PLANS.md：状态不明确时等下一帧）。
2. **分母为 0 时写 0**：除数缺失（如总时长为 0、冷却上限为 0）时不做除法，直接写 0。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_ENEMIES = 20
MAX_OPERATORS = 12
ENEMY_FEATURES = 6
OPERATOR_FEATURES = 4
GLOBAL_FEATURES = 4

# 归一化分母（PLANS.md 状态向量定义）。
COST_DIVISOR = 99.0
LEAK_DIVISOR = 3.0
ENEMY_REMAINING_DIVISOR = 20.0

#: 状态向量总维度：120 + 1 + 48 + 4 = 173。
STATE_DIM = MAX_ENEMIES * ENEMY_FEATURES + 1 + MAX_OPERATORS * OPERATOR_FEATURES + GLOBAL_FEATURES


def _clamp01(value: float) -> float:
    """把任意浮点钳到 [0, 1]；NaN 按 0 处理（不把 NaN 灌进网络）。"""

    if value != value:  # NaN
        return 0.0
    return min(1.0, max(0.0, value))


def _ratio(numerator: float, denominator: float) -> float:
    """分母为 0（或负）时返回 0，否则返回钳到 [0, 1] 的比值。"""

    if denominator <= 0:
        return 0.0
    return _clamp01(numerator / denominator)


class EnemySlot(BaseModel):
    """一个敌人槽位：`x, y` 是归一化中心，`w, h` 是归一化宽高。"""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)
    is_elite: bool = False
    alive: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="仅用于超 20 槽时排序")


class OperatorSlot(BaseModel):
    """一个干员槽位（占位按部署栏顺序）。"""

    model_config = ConfigDict(frozen=True)

    deployed: bool = False
    cooldown_remaining: float = Field(default=0.0, ge=0.0)
    cooldown_total: float = Field(default=0.0, ge=0.0)
    cost: int = Field(default=0, ge=0)
    skill_ready: bool = False


class GlobalState(BaseModel):
    """关卡级信息：进度、击杀、漏怪与剩余敌人估计。"""

    model_config = ConfigDict(frozen=True)

    progress_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)
    kills: int = Field(default=0, ge=0)
    kill_target: int = Field(default=0, ge=0)
    leaks: int = Field(default=0, ge=0)
    enemies_remaining: int = Field(default=0, ge=0)


class BattleState(BaseModel):
    """一帧战场状态；`to_vector()` 输出的就是决策层的输入。"""

    model_config = ConfigDict(frozen=True)

    enemies: tuple[EnemySlot, ...] = ()
    cost: int | None = Field(default=None, ge=0)
    operators: tuple[OperatorSlot, ...] = ()
    global_state: GlobalState = Field(default_factory=GlobalState)

    def enemy_slots(self) -> list[EnemySlot]:
        """超过 20 个时按检测置信度取前 20（PLANS.md：补零与截断规则）。"""

        ordered = sorted(self.enemies, key=lambda slot: slot.confidence, reverse=True)
        return ordered[:MAX_ENEMIES]

    def to_vector(self) -> list[float]:
        """展开成定长 173 维向量，元素全部落在 [0, 1]。"""

        vector: list[float] = []

        slots = self.enemy_slots()
        for slot in slots:
            vector.extend(
                [
                    _clamp01(slot.x),
                    _clamp01(slot.y),
                    _clamp01(slot.w),
                    _clamp01(slot.h),
                    1.0 if slot.is_elite else 0.0,
                    1.0 if slot.alive else 0.0,
                ]
            )
        empty_enemy = [0.0] * ENEMY_FEATURES
        for _ in range(MAX_ENEMIES - len(slots)):
            vector.extend(empty_enemy)

        vector.append(_ratio(float(self.cost or 0), COST_DIVISOR))

        operators = list(self.operators[:MAX_OPERATORS])
        for operator in operators:
            vector.extend(
                [
                    1.0 if operator.deployed else 0.0,
                    _ratio(operator.cooldown_remaining, operator.cooldown_total),
                    _ratio(float(operator.cost), COST_DIVISOR),
                    1.0 if operator.skill_ready else 0.0,
                ]
            )
        empty_operator = [0.0] * OPERATOR_FEATURES
        for _ in range(MAX_OPERATORS - len(operators)):
            vector.extend(empty_operator)

        state = self.global_state
        vector.extend(
            [
                _ratio(state.progress_s, state.total_s),
                _ratio(float(state.kills), float(state.kill_target)),
                _ratio(float(state.leaks), LEAK_DIVISOR),
                _ratio(float(state.enemies_remaining), ENEMY_REMAINING_DIVISOR),
            ]
        )

        assert len(vector) == STATE_DIM, f"状态向量维度应为 {STATE_DIM}，实际 {len(vector)}"
        return vector

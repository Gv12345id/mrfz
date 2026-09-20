"""单帧感知管线：`Perception.read(frame) -> BattleState`。

装配顺序：检测敌人 → 读费用 → 读干员卡数字 → 拼成定长状态向量。

**当前实现到什么程度（T7 骨架，2026-09-20）**

已实现：分辨率校验、限频与缓存、YOLOv8 框 → 归一化敌人槽位、费用与卡数字 OCR 接入。

还没做（等 T2 重训出可用权重后再补，见 `runs/HANDOFF.md` 的 T7）：

- 干员的「部署位是否占用 / 技能是否就绪」识别——目前只填卡面数字，`deployed` 与
  `skill_ready` 一律 False，避免猜出来的值污染决策层的动作掩码；
- PLANS.md 提到的「并行推理」——单检测器 + 单 OCR 目前是串行的，等测出 p95 再决定要不要并行；
- `read_cost_consensus`（双次读数一致才采信）需要连续两帧，属于后续的取帧循环，不在单帧接口里。

分辨率不匹配时**直接报错**，不缩放：ROI 是按 1280×720 逻辑坐标写死的，缩放会让费用 ROI
落在别的元素上（与 device 层「非 16:9 拒绝运行」同一条原则）。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from arknights_agent.imaging import Frame
from arknights_agent.perception.detector import Detector, EnemyBox
from arknights_agent.perception.ocr import (
    CARD_ROW_ROI,
    COST_ROI,
    DigitTemplates,
    read_card_numbers,
    read_cost,
)
from arknights_agent.perception.state import (
    MAX_OPERATORS,
    BattleState,
    EnemySlot,
    OperatorSlot,
)


class PerceptionError(RuntimeError):
    """感知管线错误（分辨率不符、必需依赖缺失等）。"""


def _to_enemy_slot(box: EnemyBox, frame_size: tuple[int, int]) -> EnemySlot:
    """像素框 → 归一化槽位；越界部分裁掉后再归一化。"""

    width, height = frame_size
    x1 = min(max(box.x1, 0.0), float(width)) / width
    y1 = min(max(box.y1, 0.0), float(height)) / height
    x2 = min(max(box.x2, 0.0), float(width)) / width
    y2 = min(max(box.y2, 0.0), float(height)) / height
    return EnemySlot(
        x=(x1 + x2) / 2,
        y=(y1 + y2) / 2,
        w=x2 - x1,
        h=y2 - y1,
        is_elite=box.is_elite,
        alive=True,
        confidence=min(1.0, max(0.0, box.confidence)),
    )


class Perception:
    """把一帧画面变成 `BattleState`；可选限频（同一周期内复用上一帧结果）。"""

    def __init__(
        self,
        detector: Detector,
        *,
        templates: DigitTemplates | None = None,
        logical_size: tuple[int, int] = (1280, 720),
        cost_roi: tuple[int, int, int, int] = COST_ROI,
        card_roi: tuple[int, int, int, int] = CARD_ROW_ROI,
        min_interval_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._templates = templates
        self._logical_size = logical_size
        self._cost_roi = cost_roi
        self._card_roi = card_roi
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._last_read_ts: float | None = None
        self._cached: BattleState | None = None

    def read(self, frame: Frame) -> BattleState:
        """读一帧。分辨率不等于逻辑尺寸时抛 :class:`PerceptionError`。"""

        width, height = int(frame.shape[1]), int(frame.shape[0])
        if (width, height) != self._logical_size:
            raise PerceptionError(
                f"画面尺寸 {width}x{height} 与逻辑尺寸 "
                f"{self._logical_size[0]}x{self._logical_size[1]} 不一致；ROI 按逻辑坐标写死，"
                "请在采集/缩放阶段先归一化分辨率"
            )

        now = self._clock()
        if (
            self._cached is not None
            and self._last_read_ts is not None
            and self._min_interval_s > 0
            and now - self._last_read_ts < self._min_interval_s
        ):
            return self._cached

        boxes = self._detector.detect(frame)
        enemies = tuple(_to_enemy_slot(box, (width, height)) for box in boxes)
        cost = read_cost(frame, self._templates, self._cost_roi) if self._templates else None
        operators = self._read_operators(frame)

        state = BattleState(enemies=enemies, cost=cost, operators=operators)
        self._cached = state
        self._last_read_ts = now
        return state

    def _read_operators(self, frame: Frame) -> tuple[OperatorSlot, ...]:
        """从底部卡片读部署费用，补足到 12 个槽位。

        `deployed` / `cooldown_remaining` / `skill_ready` 需要额外的界面识别，
        这里留默认值（见模块 docstring 的「还没做」）。
        """

        if self._templates is None:
            return tuple(OperatorSlot() for _ in range(MAX_OPERATORS))
        numbers = read_card_numbers(frame, self._templates, self._card_roi)
        slots = [OperatorSlot(cost=value) for value in numbers if value is not None]
        while len(slots) < MAX_OPERATORS:
            slots.append(OperatorSlot())
        return tuple(slots[:MAX_OPERATORS])

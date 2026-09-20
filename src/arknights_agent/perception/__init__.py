"""感知层：截图 → 结构化状态向量。"""

from arknights_agent.perception.detector import (
    Detector,
    DetectorError,
    EnemyBox,
    YoloDetector,
    enemy_boxes,
)
from arknights_agent.perception.ocr import (
    COST_ROI,
    DigitTemplates,
    OcrError,
    extract_glyphs,
    read_cost,
    read_cost_consensus,
)
from arknights_agent.perception.pipeline import Perception, PerceptionError
from arknights_agent.perception.state import (
    MAX_ENEMIES,
    MAX_OPERATORS,
    STATE_DIM,
    BattleState,
    EnemySlot,
    GlobalState,
    OperatorSlot,
)

__all__ = [
    "COST_ROI",
    "MAX_ENEMIES",
    "MAX_OPERATORS",
    "STATE_DIM",
    "BattleState",
    "Detector",
    "DetectorError",
    "DigitTemplates",
    "EnemyBox",
    "EnemySlot",
    "GlobalState",
    "OcrError",
    "OperatorSlot",
    "Perception",
    "PerceptionError",
    "YoloDetector",
    "enemy_boxes",
    "extract_glyphs",
    "read_cost",
    "read_cost_consensus",
]

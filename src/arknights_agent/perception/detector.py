"""YOLOv8 敌人检测封装：`detect(frame) -> list[EnemyBox]`。

分工：

- **纯逻辑**（:func:`enemy_boxes`）：把推理输出的原始行按置信度过滤、裁到画面内、按面积排序。
  不依赖 torch，可以直接单测。
- **胶水**（:class:`YoloDetector`）：加载 `models/yolov8n_arknights.pt` 并调用 ultralytics。
  `ultralytics` 在构造时才 import，避免离线单测被 torch 拖慢。

阈值与权重路径来自配置（`config/config.yaml` 的 `perception` 段），不写死在调用处。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from arknights_agent.imaging import Frame

#: 敌人只有一个类别（PLANS.md：类别仅区分普通 / 精英时按 class_id 映射）。
ENEMY_CLASS_ID = 0
ELITE_CLASS_ID = 1


class DetectorError(RuntimeError):
    """检测器初始化或推理失败（权重缺失、依赖缺失等）。"""


class Detector(Protocol):
    """检测器接口；离线回放与真机共用同一个接口（AGENTS.md：跨层接口用 Protocol）。"""

    def detect(self, frame: Frame) -> list[EnemyBox]:
        """返回这一帧里的敌人框（像素坐标）。"""
        ...


@dataclass(frozen=True)
class EnemyBox:
    """一个敌人检测框（像素坐标，左上角 + 右下角）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = ENEMY_CLASS_ID

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_elite(self) -> bool:
        return self.class_id == ELITE_CLASS_ID


def _clip_box(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """把框裁进画面；裁完宽或高不到 1px 就丢掉（返回 None）。"""

    x1 = min(max(xyxy[0], 0.0), float(width))
    y1 = min(max(xyxy[1], 0.0), float(height))
    x2 = min(max(xyxy[2], 0.0), float(width))
    y2 = min(max(xyxy[3], 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def enemy_boxes(
    rows: Iterable[Sequence[float]],
    *,
    frame_size: tuple[int, int],
    min_confidence: float,
) -> list[EnemyBox]:
    """原始检测行 → 敌人框列表。

    `rows` 每行是 `(x1, y1, x2, y2, confidence, class_id)`。低于 `min_confidence`
    的行丢掉；越界框裁进画面；结果按面积从大到小排序（近处的敌人优先占槽位）。
    """

    width, height = frame_size
    boxes: list[EnemyBox] = []
    for row in rows:
        x1, y1, x2, y2, confidence, class_id = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            int(row[5]),
        )
        if confidence < min_confidence:
            continue
        clipped = _clip_box((x1, y1, x2, y2), width, height)
        if clipped is None:
            continue
        boxes.append(
            EnemyBox(
                x1=clipped[0],
                y1=clipped[1],
                x2=clipped[2],
                y2=clipped[3],
                confidence=confidence,
                class_id=class_id,
            )
        )
    boxes.sort(key=lambda box: box.area, reverse=True)
    return boxes


def _rows_from_result(result: object) -> list[list[float]]:
    """从 ultralytics 的单个 `Results` 里取出 `(x1,y1,x2,y2,conf,cls)` 行。"""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coordinates = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    classes = boxes.cls.tolist()
    return [
        [*coordinate, confidence, class_id]
        for coordinate, confidence, class_id in zip(coordinates, confidences, classes, strict=True)
    ]


class YoloDetector:
    """Ultralytics YOLOv8 推理封装；一次只跑一张图（batch=1，CPU 基准）。"""

    def __init__(
        self,
        weights: Path,
        *,
        min_confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        if not weights.is_file():
            raise DetectorError(
                f"检测权重不存在：{weights}；先用 tools/train_yolo.py 训练并归档到 models/"
            )
        try:
            # ultralytics 的 __init__ 没有显式 re-export YOLO，mypy strict 会报 attr-defined；
            # 这里是第三方库的既有行为，用局部 ignore 而不是给整个包关掉类型检查。
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可执行的建议
            raise DetectorError(
                '未安装 ultralytics；执行 `pip install -e ".[dev]"` 或 `pip install ultralytics`'
            ) from exc
        self._weights = weights
        self._min_confidence = min_confidence
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._model = YOLO(str(weights))

    @property
    def weights(self) -> Path:
        return self._weights

    def detect(self, frame: Frame) -> list[EnemyBox]:
        """跑一帧检测；返回按面积降序的敌人框。"""

        results = self._model.predict(
            source=frame,
            conf=self._min_confidence,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            device="cpu",
            verbose=False,
        )
        rows = [row for result in results for row in _rows_from_result(result)]
        height, width = frame.shape[:2]
        return enemy_boxes(
            rows, frame_size=(int(width), int(height)), min_confidence=self._min_confidence
        )

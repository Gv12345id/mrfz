"""在 holdout 集上评估感知层，产出 `runs/phase2/perception_metrics.json`。

这个文件是 `scripts/verify.py --phase 2` 的输入：`enemy-map50`、`enemy-recall`、
`state-dim`、`perception-latency-p95-ms` 四个 check 都读它。阈值写在门禁里，
本工具**只产出数字、不判通过**——判通过是门禁的事。

字段口径（与 PLANS.md 阶段 2 验收 2/4 对应）：

- `enemy_map50` / `enemy_map50_95` / `enemy_box_precision` / `enemy_box_recall`：
  Ultralytics 在 holdout 上的框级指标；
- `enemy_recall`：**帧级「战场存在敌人」召回**——真值有敌人的帧里，模型没把整帧判空的比例
  （PLANS.md 写的是「『战场存在敌人』召回 ≥ 0.90」，指的就是这一条）；
- `state_dim`：跑一遍 `Perception.read()`，确认状态向量维度；
- `latency_p95_ms`：单帧 `Perception.read()` 的 p95（最近秩法，与 verify.py 的 p95 同口径），
  CPU、batch=1。

holdout 由 `tools/build_yolo_dataset.py --holdout-blocks` 产出，**永不参与训练与早停**，
所以这里的数字才是独立的。

用法::

    .\\.venv\\Scripts\\python.exe tools\\eval_perception.py \\
        --weights models/yolov8n_arknights.pt \\
        --data runs/phase2/yolo_dataset_v2/data_holdout.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from arknights_agent.imaging import read_bgr
from arknights_agent.perception.detector import YoloDetector
from arknights_agent.perception.ocr import DEFAULT_TEMPLATE_PATH, DigitTemplates, OcrError
from arknights_agent.perception.pipeline import Perception
from arknights_agent.perception.state import STATE_DIM

LOCAL_TZ = timezone(timedelta(hours=8))
DEFAULT_REPORT = Path("runs/phase2/perception_metrics.json")
DEFAULT_TEMPLATES = DEFAULT_TEMPLATE_PATH


def p95_ms(samples_ms: Sequence[float]) -> float:
    """最近秩法 p95：升序后取 `ceil(0.95 * n)` 位，与 verify.py 同口径。"""

    if not samples_ms:
        raise ValueError("样本为空，无法计算 p95")
    ordered = sorted(samples_ms)
    rank = min(len(ordered), max(1, math.ceil(0.95 * len(ordered))))
    return round(float(ordered[rank - 1]), 3)


def presence_scores(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    """帧级「有没有敌人」的混淆统计。

    召回低说明模型把有敌人的帧整帧判空（决策层会以为安全）——这是比框不准更危险的一类错。
    """

    if len(truth) != len(predicted):
        raise ValueError(f"真值 {len(truth)} 帧与预测 {len(predicted)} 帧数量不一致")
    if not truth:
        raise ValueError("样本为空，无法计算帧级指标")
    tp = sum(1 for expect, got in zip(truth, predicted, strict=True) if expect and got)
    fn = sum(1 for expect, got in zip(truth, predicted, strict=True) if expect and not got)
    fp = sum(1 for expect, got in zip(truth, predicted, strict=True) if not expect and got)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "frames": len(truth),
        "frames_with_enemy": tp + fn,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def label_has_box(label_path: Path) -> bool:
    """YOLO 标签文件里是否有至少一个框（空文件 = 该帧没有敌人）。"""

    if not label_path.is_file():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def metric_value(metrics: Any, key: str) -> float | None:
    """从 Ultralytics 的 metrics 对象里取指标（版本间字段名略有差异）。"""

    box = getattr(metrics, "box", None)
    for source in (box, metrics, getattr(metrics, "results_dict", None)):
        if source is None:
            continue
        value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        if value is not None:
            return round(float(value), 4)
    return None


def load_templates(path: Path | None) -> DigitTemplates | None:
    """有模板就用，没有就退化成"只测检测"并如实记录（不假装 OCR 测过）。"""

    if path is None:
        return None
    try:
        return DigitTemplates.load(path)
    except OcrError:
        return None


def evaluate(
    weights: Path,
    data_yaml: Path,
    *,
    report: Path = DEFAULT_REPORT,
    conf: float = 0.25,
    latency_samples: int = 50,
    templates_path: Path | None = DEFAULT_TEMPLATES,
) -> dict[str, Any]:
    """跑 holdout 评估并落盘报告。"""

    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}；先跑 tools/train_yolo.py")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在：{data_yaml}")

    holdout_dir = data_yaml.parent
    images_dir = holdout_dir / "images" / "holdout"
    labels_dir = holdout_dir / "labels" / "holdout"
    frames = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"{images_dir} 里没有 holdout 帧")

    detector = YoloDetector(weights, min_confidence=conf)

    # 1) 框级指标：直接让 ultralytics 在 holdout 上算 mAP。
    box_metrics = YOLO(str(weights)).val(
        data=str(data_yaml), imgsz=640, device="cpu", conf=conf, verbose=False
    )

    # 2) 帧级「有没有敌人」：逐帧比对真值标签与检测结果。
    truth: list[bool] = []
    predicted: list[bool] = []
    for image_path in frames:
        truth.append(label_has_box(labels_dir / f"{image_path.stem}.txt"))
        predicted.append(bool(detector.detect(read_bgr(image_path))))
    presence = presence_scores(truth, predicted)

    # 3) 状态向量维度与单帧延迟：走真实的 Perception.read 路径。
    templates = load_templates(templates_path)
    perception = Perception(detector, templates=templates)
    sample_frames = frames[: max(1, min(latency_samples, len(frames)))]
    durations: list[float] = []
    state_dim = 0
    for image_path in sample_frames:
        frame = read_bgr(image_path)
        started = time.perf_counter()
        state = perception.read(frame)
        durations.append((time.perf_counter() - started) * 1000)
        state_dim = len(state.to_vector())

    payload: dict[str, Any] = {
        "artifact": "perception_metrics",
        "created_at": datetime.now(LOCAL_TZ).isoformat(),
        "weights": str(weights).replace("\\", "/"),
        "data_yaml": str(data_yaml).replace("\\", "/"),
        "holdout_frames": len(frames),
        "detector_conf": conf,
        "templates": str(templates_path).replace("\\", "/") if templates else None,
        "enemy_map50": metric_value(box_metrics, "map50"),
        "enemy_map50_95": metric_value(box_metrics, "map"),
        "enemy_box_precision": metric_value(box_metrics, "mp"),
        "enemy_box_recall": metric_value(box_metrics, "mr"),
        "enemy_recall": presence["recall"],
        "enemy_presence_precision": presence["precision"],
        "presence_detail": presence,
        "state_dim": state_dim,
        "state_dim_expected": STATE_DIM,
        "latency_samples": len(durations),
        "latency_p95_ms": p95_ms(durations),
        "latency_mean_ms": round(sum(durations) / len(durations), 3),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="holdout 感知评估 → perception_metrics.json")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n_arknights.pt"))
    parser.add_argument(
        "--data", type=Path, default=Path("runs/phase2/yolo_dataset_v2/data_holdout.yaml")
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATES,
        help="数字模板 npz；不存在时只测检测，报告里 templates 记 null",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = evaluate(
        args.weights,
        args.data,
        report=args.report,
        conf=args.conf,
        latency_samples=args.latency_samples,
        templates_path=args.templates,
    )
    print(
        f"holdout {payload['holdout_frames']} 帧：mAP@0.5={payload['enemy_map50']}｜"
        f"框召回={payload['enemy_box_recall']}｜帧级召回={payload['enemy_recall']}｜"
        f"state_dim={payload['state_dim']}｜p95={payload['latency_p95_ms']}ms"
    )
    print(f"报告：{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""阶段 2 步骤 3：用标注数据训练 YOLOv8n 并报告留出集指标。

产物：

- Ultralytics 训练目录 `runs/phase2/yolo_runs/<name>/`；
- `models/yolov8n_arknights.pt`：best.pt 的副本（权重不入库，`.gitignore` 挡住）；
- `runs/phase2/training_report.json`：训练时长、mAP@0.5、precision、recall 等指标。

用法::

    .\\.venv\\Scripts\\python.exe tools\\train_yolo.py \\
        --data runs/phase2/yolo_dataset/data.yaml --epochs 60 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOCAL_TZ = timezone(timedelta(hours=8))
DEFAULT_BASE_WEIGHTS = "yolov8n.pt"
DEFAULT_MODEL_OUT = Path("models/yolov8n_arknights.pt")
DEFAULT_REPORT = Path("runs/phase2/training_report.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv8n 并评估")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml 路径")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12, help="torch CPU 线程数")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-weights", default=DEFAULT_BASE_WEIGHTS, help="预训练权重或空字符串")
    parser.add_argument("--project", type=Path, default=Path("runs/phase2/yolo_runs"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _metric(metrics: Any, key: str) -> float | None:
    """从 Ultralytics 的 metrics 对象里取指标（不同版本字段名略有差异）。"""

    box = getattr(metrics, "box", None)
    for source in (box, metrics, getattr(metrics, "results_dict", None)):
        if source is None:
            continue
        value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        if value is not None:
            return float(value)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch  # 延迟导入：不训练时不加载 torch
    from ultralytics import YOLO  # 延迟导入：不训练时不加载 torch

    # 不显式设置时，导入 ultralytics 之后 torch 线程数会被压到 1，CPU 训练会慢一个数量级。
    torch.set_num_threads(max(1, args.threads))
    started_at = datetime.now(LOCAL_TZ)
    started = time.perf_counter()
    model = YOLO(args.base_weights) if args.base_weights else YOLO("yolov8n.yaml")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    train_seconds = time.perf_counter() - started

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, args.model_out)

    tuned = YOLO(str(args.model_out))
    metrics = tuned.val(data=str(args.data), imgsz=args.imgsz, device=args.device, verbose=False)
    report = {
        "artifact": "yolo_training",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(LOCAL_TZ).isoformat(),
        "train_seconds": round(train_seconds, 1),
        "train_minutes": round(train_seconds / 60, 2),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "device": args.device,
        "threads": args.threads,
        "base_weights": args.base_weights,
        "data": str(args.data).replace("\\", "/"),
        "best_weights": str(best).replace("\\", "/"),
        "model_out": str(args.model_out).replace("\\", "/"),
        "map50": _metric(metrics, "map50"),
        "map50_95": _metric(metrics, "map"),
        "precision": _metric(metrics, "mp"),
        "recall": _metric(metrics, "mr"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"训练完成：{report['train_minutes']} 分钟｜mAP@0.5={report['map50']}｜"
        f"precision={report['precision']}｜recall={report['recall']}"
    )
    print(f"权重：{args.model_out}｜报告：{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

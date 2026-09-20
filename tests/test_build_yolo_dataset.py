"""`tools/build_yolo_dataset.py` 的纯逻辑测试（不训练、不联网）。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BUILD_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_yolo_dataset.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_yolo_dataset", BUILD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def test_to_yolo_lines_normalizes_and_skips_ignored() -> None:
    boxes = [
        builder.Box(0.0, 0.0, 128.0, 144.0),
        builder.Box(100.0, 100.0, 200.0, 200.0, ignored=True),
    ]

    lines = builder.to_yolo_lines(boxes, width=1280, height=720)

    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    assert abs(float(cx) - 0.05) < 1e-6
    assert abs(float(cy) - 0.1) < 1e-6
    assert abs(float(w) - 0.1) < 1e-6
    assert abs(float(h) - 0.2) < 1e-6


def test_to_yolo_lines_clamps_out_of_range_and_drops_degenerate() -> None:
    boxes = [
        builder.Box(-50.0, -50.0, 100.0, 100.0),
        builder.Box(1000.0, 1000.0, 1001.0, 1001.0),  # 裁完退化
    ]

    lines = builder.to_yolo_lines(boxes, width=1280, height=720)

    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_split_block_indices_is_contiguous_and_disjoint() -> None:
    train, val = builder.split_block_indices(100, blocks=5, val_blocks=(4,))

    assert train == list(range(80))
    assert val == list(range(80, 100))
    assert set(train).isdisjoint(val)


def test_build_excludes_ignored_and_writes_split(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    records = []
    for step in range(1, 11):
        path = frames / f"frame_{step:04d}.png"
        cv2.imwrite(str(path), np.zeros((720, 1280, 3), dtype=np.uint8))
        labels = [
            {"xyxy": [100, 100, 200, 200], "class_id": 0, "class_name": "enemy", "source": "auto"}
        ]
        if step == 1:
            labels.append(
                {
                    "xyxy": [300, 300, 400, 400],
                    "class_id": 0,
                    "class_name": "enemy",
                    "source": "auto",
                    "ignored": True,
                }
            )
        records.append(
            {
                "index": step,
                "step": step,
                "result": "ok",
                "batch": "b1",
                "screenshot": str(path).replace("\\", "/"),
                "labels": labels,
            }
        )
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"records": records}), encoding="utf-8")

    stats = builder.build(annotation, tmp_path / "ds", blocks=5, val_blocks=(4,))

    assert stats["ignored_boxes_skipped"] == 1
    assert stats["splits"]["train"]["frames"] == 8
    assert stats["splits"]["val"]["frames"] == 2
    first_label = tmp_path / "ds" / "labels" / "train" / "i0001.txt"
    assert len(first_label.read_text(encoding="utf-8").splitlines()) == 1  # ignored 被排除
    assert (tmp_path / "ds" / "data.yaml").is_file()

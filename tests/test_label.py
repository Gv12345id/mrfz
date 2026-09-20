"""`tools/label.py` 的纯逻辑测试（不启动 GUI、不需要权重）。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LABEL_PATH = Path(__file__).resolve().parents[1] / "tools" / "label.py"


def _load_label() -> Any:
    spec = importlib.util.spec_from_file_location("label_tool", LABEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


label = _load_label()


def test_key_action_maps_required_shortcuts() -> None:
    """用户要求的五个快捷键必须都有映射。"""

    assert label.key_action("Left") == "prev_frame"
    assert label.key_action("Right") == "next_frame"
    assert label.key_action("x") == "delete_box"
    assert label.key_action("r") == "toggle_draw"
    assert label.key_action("s") == "save"
    assert label.key_action("F5") is None


def test_make_label_clamps_to_frame_and_keeps_xyxy() -> None:
    box = label.make_label([-10, -20, 1300, 800], source="human")

    assert box["xyxy"] == [0.0, 0.0, 1279.0, 719.0]
    assert box["class_id"] == 0
    assert box["class_name"] == "enemy"
    assert box["source"] == "human"
    assert "score" not in box


def test_merge_labels_drops_overlapping_duplicates() -> None:
    a = label.make_label([100, 100, 160, 160], source="auto")
    b = label.make_label([104, 104, 164, 164], source="auto")
    c = label.make_label([400, 300, 460, 360], source="auto")

    merged = label.merge_labels([a, b, c])

    assert len(merged) == 2
    assert merged[0]["xyxy"] == a["xyxy"]


def test_heuristic_labels_finds_moving_blob_only() -> None:
    background = np.full((720, 1280, 3), 40, dtype=np.uint8)
    before = background.copy()
    after = background.copy()
    # 画面中央放一个"敌人"，只改 after，制造帧间差分
    cv2.rectangle(after, (600, 300), (660, 360), (200, 200, 200), -1)
    # UI 屏蔽区里的变化（底部干员栏）不应被当成敌人
    cv2.rectangle(after, (600, 640), (660, 700), (200, 200, 200), -1)

    labels = label.heuristic_labels(before, after)

    assert len(labels) == 1
    x1, y1, x2, y2 = labels[0]["xyxy"]
    assert 590 <= x1 <= 610 and 290 <= y1 <= 310
    assert x2 - x1 <= label.MAX_BOX_SIDE and y2 - y1 <= label.MAX_BOX_SIDE


def test_pre_annotate_heuristic_writes_labels_and_skips_bad_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    background = np.full((720, 1280, 3), 30, dtype=np.uint8)
    records = []
    for step in range(1, 4):
        frame = background.copy()
        cv2.rectangle(frame, (500 + step * 20, 300), (560 + step * 20, 360), (220, 220, 220), -1)
        path = frames_dir / f"frame_{step:04d}.png"
        cv2.imwrite(str(path), frame)
        records.append(
            {
                "step": step,
                "result": "ok",
                "screenshot": str(path).replace("\\", "/"),
                "labels": [],
            }
        )
    records.append({"step": 4, "result": "skip:dark", "screenshot": "", "labels": []})
    data = {"artifact": "annotation", "records": records}

    stats = label.pre_annotate(data, mode="heuristic")

    assert stats["frames"] == 3
    assert stats["labels"] >= 1
    assert data["records"][0]["labels"] == []  # 第一帧没有前一帧可比
    assert data["records"][1]["labels"][0]["source"] == "auto"
    assert data["records"][3]["labels"] == []
    json.dumps(data)  # 必须可序列化

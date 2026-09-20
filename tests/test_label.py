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
    assert label.key_action("e") == "toggle_ignored"
    assert label.key_action("t") == "toggle_ignored_neighbours"
    assert label.key_action("r") == "toggle_draw"
    assert label.key_action("s") == "save"
    assert label.key_action("F5") is None


def test_glow_fraction_flags_synthetic_glow() -> None:
    """高亮高饱和的方块应被判为发光特效，普通暗色纹理不会。"""

    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    box = [500, 300, 560, 360]
    assert label.is_glow_like(frame, box) is False

    glow = frame.copy()
    glow[300:360, 500:560] = (255, 255, 255)  # 纯白高亮 → V 高、S 低
    assert label.is_glow_like(glow, box) is False  # 白色是低饱和，不算发光

    colored_glow = frame.copy()
    colored_glow[300:360, 500:560] = (255, 40, 40)  # 蓝通道高值 + 饱和 (BGR)
    assert label.is_glow_like(colored_glow, box) is True


def test_heuristic_labels_drops_glow_blob() -> None:
    """移动的发光特效不应被框出来，移动的"实体"仍然保留。"""

    background = np.full((720, 1280, 3), 40, dtype=np.uint8)
    before = background.copy()
    after = background.copy()
    # 发光特效：亮蓝 + 高饱和
    cv2.rectangle(after, (300, 300), (360, 360), (255, 60, 60), -1)
    # 实体：中灰 + 深色描边
    cv2.rectangle(after, (700, 300), (760, 360), (150, 150, 150), -1)
    cv2.rectangle(after, (700, 300), (760, 360), (30, 30, 30), 3)

    labels = label.heuristic_labels(before, after)

    assert len(labels) == 1
    x1, _y1, _x2, _y2 = labels[0]["xyxy"]
    assert 680 <= x1 <= 720  # 只留下那个实体方块


def test_static_filter_drops_unchanged_region(tmp_path: Path) -> None:
    """连续多帧像素不变的框（地面装饰）会被丢掉，有变化的框保留。"""

    base = np.full((720, 1280, 3), 40, dtype=np.uint8)
    records = []
    for step in range(5):
        frame = base.copy()
        # 静态装饰：四帧完全一样
        cv2.rectangle(frame, (300, 300), (360, 360), (90, 90, 90), -1)
        # 会动的目标：每帧颜色都变
        cv2.rectangle(frame, (700, 300), (760, 360), (90 + step * 20,) * 3, -1)
        path = tmp_path / f"f{step}.png"
        cv2.imwrite(str(path), frame)
        records.append(
            {
                "step": step,
                "result": "ok",
                "batch": "b",
                "screenshot": str(path).replace("\\", "/"),
                "labels": [
                    label.make_label([300, 300, 360, 360], source="auto"),
                    label.make_label([700, 300, 760, 360], source="auto"),
                ],
            }
        )
    data = {"records": records}

    stats = label.static_filter(data)

    assert stats["removed"] >= 1
    assert all(len(item["labels"]) == 1 for item in records)
    assert records[0]["labels"][0]["xyxy"][0] == 700.0


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
    # a 与 b 重叠，合并成外接矩形；排序规则是先上后下、先左后右
    assert [item["xyxy"] for item in merged] == [[100.0, 100.0, 164.0, 164.0], c["xyxy"]]


def test_merge_labels_merges_shadow_box_by_center_distance() -> None:
    """敌人快速移动留下的"影子框"：几乎不重叠但中心很近，应合成一个外接矩形。"""

    a = label.make_label([600, 300, 660, 360], source="auto")
    b = label.make_label([640, 300, 700, 360], source="auto")

    merged = label.merge_labels([a, b])

    assert len(merged) == 1
    assert merged[0]["xyxy"] == [600.0, 300.0, 700.0, 360.0]


def test_merge_labels_merges_contained_box() -> None:
    """大框套小框（IoU 低、包含率高）也应合并，而不是留两个重叠框。"""

    big = label.make_label([500, 280, 620, 400], source="auto")
    small = label.make_label([540, 320, 580, 360], source="auto")

    merged = label.merge_labels([big, small])

    assert len(merged) == 1
    assert merged[0]["xyxy"] == big["xyxy"]


def test_merge_labels_caps_count_and_sorts() -> None:
    boxes = [
        label.make_label(
            [100 + 150 * i, 200 + (i % 2) * 120, 160 + 150 * i, 260 + (i % 2) * 120],
            source="auto",
        )
        for i in range(9)
    ]

    merged = label.merge_labels(boxes)

    assert len(merged) == label.MAX_LABELS
    ys = [item["xyxy"][1] for item in merged]
    assert ys == sorted(ys)


def test_align_frame_compensates_known_translation() -> None:
    """整幅画面平移 12px 时，补偿后残差应远小于未补偿。"""

    base = np.zeros((720, 1280), dtype=np.uint8)
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, size=(500, 900), dtype=np.uint8)
    base[100:600, 150:1050] = patch
    matrix = np.float32([[1, 0, 12], [0, 1, 9]])
    shifted = cv2.warpAffine(base, matrix, (1280, 720), borderMode=cv2.BORDER_REPLICATE)

    aligned, shift = label.align_frame(base, shifted)

    raw_residual = float(np.mean(cv2.absdiff(base, shifted)))
    aligned_residual = float(np.mean(cv2.absdiff(aligned, shifted)))
    assert aligned_residual < raw_residual * 0.5
    assert abs(shift[0] - 12) < 2 and abs(shift[1] - 9) < 2


def test_heuristic_labels_ignores_pure_camera_pan() -> None:
    """只有镜头平移、没有独立运动目标时，不应产出框。"""

    base = np.zeros((720, 1280, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    patch = rng.integers(0, 255, size=(500, 900, 3), dtype=np.uint8)
    base[100:600, 150:1050] = patch
    matrix = np.float32([[1, 0, 14], [0, 1, 10]])
    shifted = cv2.warpAffine(base, matrix, (1280, 720), borderMode=cv2.BORDER_REPLICATE)

    labels = label.heuristic_labels(base, shifted)

    assert labels == []


def test_temporal_filter_drops_single_frame_blips_and_keeps_stable_boxes() -> None:
    """连帧稳定的框保留；只闪一帧的特效框丢掉；人工框无条件保留。"""

    def box(x: int, y: int) -> dict:
        return label.make_label([x, y, x + 40, y + 40], source="auto")

    stable = box(400, 300)
    records = [
        {"step": 1, "result": "ok", "batch": "b1", "labels": [stable]},
        {"step": 2, "result": "ok", "batch": "b1", "labels": [box(410, 300), box(900, 200)]},
        {"step": 3, "result": "ok", "batch": "b1", "labels": [box(420, 305)]},
    ]
    human = label.make_label([100, 100, 140, 140], source="human")
    records[1]["labels"].append(human)

    stats = label.temporal_filter({"records": records})

    assert stats["removed"] == 1  # 只有 (900,200) 那个单帧闪点被丢
    assert len(records[1]["labels"]) == 2
    assert any(item["source"] == "human" for item in records[1]["labels"])


def test_temporal_filter_never_compares_across_batches() -> None:
    """跨批次的时间不连续，不能拿另一批的框当邻居。"""

    def box(x: int, y: int) -> dict:
        return label.make_label([x, y, x + 40, y + 40], source="auto")

    records = [
        {"step": 1, "result": "ok", "batch": "b1", "labels": [box(400, 300)]},
        {"step": 2, "result": "ok", "batch": "b2", "labels": [box(405, 300)]},
        {"step": 3, "result": "ok", "batch": "b2", "labels": [box(410, 300)]},
    ]

    stats = label.temporal_filter({"records": records})

    assert stats["removed"] == 1  # 第 1 帧在 b1 里没有同批邻居
    assert len(records[0]["labels"]) == 0


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

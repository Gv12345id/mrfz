"""把标注 JSON 转成 Ultralytics YOLO 数据集目录。

要点：

- **排除 `ignored` 框**：人工标成"非敌人/忽略"的区域不进标签文件（等价于硬负样本），
  否则模型会学着去框特效。
- **按时间块切分**：相邻帧几乎一模一样（采集间隔 80–170ms），随机切分会让验证集和
  训练集"泄漏"。这里按批次内部顺序切成若干连续块，取最后一块做验证集。
- 没有框的帧也写进数据集（背景样本），标签文件为空。
- 图片用硬链接（同盘零拷贝），跨盘自动退回复制。

用法::

    .\\.venv\\Scripts\\python.exe tools\\build_yolo_dataset.py \\
        --annotation runs/phase2/annotation_400.json --out runs/phase2/yolo_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASS_NAMES = ["enemy"]
DEFAULT_BLOCKS = 5
DEFAULT_VAL_BLOCKS = (4,)


@dataclass(frozen=True)
class Box:
    """一个标注框（像素 xyxy）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    ignored: bool = False


def parse_box(raw: dict[str, Any]) -> Box:
    """把 JSON 里的框转成 :class:`Box`。"""

    x1, y1, x2, y2 = (float(v) for v in raw["xyxy"])
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, ignored=bool(raw.get("ignored", False)))


def to_yolo_lines(boxes: Sequence[Box], *, width: int, height: int, class_id: int = 0) -> list[str]:
    """转成 YOLO 标签行 `class cx cy w h`（归一化到 0–1），并裁掉越界与退化框。"""

    lines: list[str] = []
    for box in boxes:
        if box.ignored:
            continue
        x1 = min(max(box.x1, 0.0), float(width))
        x2 = min(max(box.x2, 0.0), float(width))
        y1 = min(max(box.y1, 0.0), float(height))
        y2 = min(max(box.y2, 0.0), float(height))
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 2 or box_h < 2:
            continue  # 退化框（被裁没了）直接丢
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box_w / width:.6f} {box_h / height:.6f}")
    return lines


def split_block_indices(
    total: int, *, blocks: int = DEFAULT_BLOCKS, val_blocks: Sequence[int] = DEFAULT_VAL_BLOCKS
) -> tuple[list[int], list[int]]:
    """按连续块切分：返回 (训练下标, 验证下标)。块是连续区间，避免相邻帧泄漏。"""

    if total <= 0:
        return [], []
    blocks = max(2, min(blocks, total))
    size = total / blocks
    train: list[int] = []
    val: list[int] = []
    for index in range(total):
        block = min(blocks - 1, int(index / size))
        (val if block in set(val_blocks) else train).append(index)
    return train, val


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def build(
    annotation: Path,
    out_dir: Path,
    *,
    blocks: int = DEFAULT_BLOCKS,
    val_blocks: Sequence[int] = DEFAULT_VAL_BLOCKS,
) -> dict[str, Any]:
    """执行转换，返回统计信息。"""

    data = json.loads(annotation.read_text(encoding="utf-8"))
    records = [item for item in data["records"] if item.get("result") == "ok"]

    # 按批次分组，保持批次内原始顺序（时间连续）
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("batch", "single")), []).append(record)

    stats: dict[str, Any] = {
        "annotation": str(annotation).replace("\\", "/"),
        "blocks": blocks,
        "val_blocks": list(val_blocks),
        "splits": {},
        "ignored_boxes_skipped": 0,
        "boxes": {"train": 0, "val": 0},
    }

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        stats["splits"][split] = {"frames": 0, "empty_frames": 0}

    for items in grouped.values():
        train_idx, val_idx = split_block_indices(len(items), blocks=blocks, val_blocks=val_blocks)
        for split, indices in (("train", train_idx), ("val", val_idx)):
            for position in indices:
                record = items[position]
                image_path = Path(record["screenshot"])
                if not image_path.is_file():
                    continue
                boxes = [parse_box(item) for item in record.get("labels") or []]
                stats["ignored_boxes_skipped"] += sum(1 for item in boxes if item.ignored)
                height, width = 720, 1280
                lines = to_yolo_lines(boxes, width=width, height=height)

                stem = f"i{record.get('index', position):04d}"
                _link(image_path, out_dir / "images" / split / f"{stem}{image_path.suffix}")
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                stats["splits"][split]["frames"] += 1
                stats["boxes"][split] += len(lines)
                if not lines:
                    stats["splits"][split]["empty_frames"] += 1

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    stats["data_yaml"] = str(yaml_path).replace("\\", "/")
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="annotation JSON → YOLO 数据集")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/phase2/yolo_dataset"))
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--val-blocks", default="4", help="用哪些连续块做验证集（逗号分隔，默认最后一块）"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    val_blocks = tuple(int(item) for item in args.val_blocks.split(",") if item.strip())
    stats = build(args.annotation, args.out, blocks=args.blocks, val_blocks=val_blocks)
    print(
        f"数据集已生成：train {stats['splits']['train']['frames']} 帧"
        f"（{stats['boxes']['train']} 框，其中空帧 {stats['splits']['train']['empty_frames']}）｜"
        f"val {stats['splits']['val']['frames']} 帧（{stats['boxes']['val']} 框）"
    )
    print(f"跳过 ignored 框 {stats['ignored_boxes_skipped']} 个｜data.yaml: {stats['data_yaml']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

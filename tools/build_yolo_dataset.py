"""把标注 JSON 转成 Ultralytics YOLO 数据集目录。

要点：

- **排除 `ignored` 框**：人工标成"非敌人/忽略"的区域不进标签文件（等价于硬负样本），
  否则模型会学着去框特效。
- **按时间块切分**：相邻帧几乎一模一样（采集间隔 80–170ms），随机切分会让验证集和
  训练集"泄漏"。这里按批次内部顺序切成若干连续块，取最后一块做验证集。
- **holdout 永不参与训练**：`--holdout-blocks` 指定的块只用于最终指标（PLANS.md 阶段 2
  验收 6）。它们既不进 `train` 也不进 `val`——早停用的 `val` 会被反复看到，
  拿它当门禁指标等于自评，所以另外写一份 `data_holdout.yaml`。
- 没有框的帧也写进数据集（背景样本），标签文件为空。
- 图片用硬链接（同盘零拷贝），跨盘自动退回复制。

用法::

    .\\.venv\\Scripts\\python.exe tools\\build_yolo_dataset.py \\
        --annotation runs/phase2/annotation_400.json --out runs/phase2/yolo_dataset

默认切分（5 块）：train = 块 0-2，val = 块 3，holdout = 块 4；996 帧下 holdout 约 199 帧。
同时把 holdout 帧链进 `tests/fixtures/perception/` 并写 `frames.sha256`
（`scripts/verify.py --phase 2` 的 `perception-fixtures-hash` 检查读的就是它）。
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_VAL_BLOCKS = (3,)
DEFAULT_HOLDOUT_BLOCKS = (4,)
DEFAULT_HOLDOUT_DIR = Path("tests/fixtures/perception")


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
    total: int,
    *,
    blocks: int = DEFAULT_BLOCKS,
    val_blocks: Sequence[int] = DEFAULT_VAL_BLOCKS,
    holdout_blocks: Sequence[int] = DEFAULT_HOLDOUT_BLOCKS,
) -> tuple[list[int], list[int], list[int]]:
    """按连续块切分：返回 (训练下标, 验证下标, holdout 下标)。

    块是连续区间（相邻帧几乎一样，必须整块切，避免训练/验证泄漏）。holdout 与 val
    互斥——同一块同时出现在两处直接报错，宁可停下也不给一个自评出来的指标。
    """

    if total <= 0:
        return [], [], []
    blocks = max(2, min(blocks, total))
    overlap = sorted(set(val_blocks) & set(holdout_blocks))
    if overlap:
        raise ValueError(f"val 块与 holdout 块重叠：{overlap}；两者必须互斥")
    size = total / blocks
    val_set = set(val_blocks)
    holdout_set = set(holdout_blocks)
    train: list[int] = []
    val: list[int] = []
    holdout: list[int] = []
    for index in range(total):
        block = min(blocks - 1, int(index / size))
        if block in holdout_set:
            holdout.append(index)
        elif block in val_set:
            val.append(index)
        else:
            train.append(index)
    if not train:
        raise ValueError("训练集为空：val_blocks 与 holdout_blocks 占满了所有块")
    return train, val, holdout


def _file_sha256(path: Path) -> str:
    """文件内容 sha256（分块读）；清单格式与 `sha256sum` 一致。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_holdout_fixtures(holdout_dir: Path, frames: Sequence[tuple[Path, int]]) -> dict[str, Any]:
    """把 holdout 帧链进 fixtures 目录，并写 `frames.sha256`。

    目录里只保留本次写入的 `frame_*.png`：上一次运行留下的旧帧会被删掉，避免
    清单与实际内容不一致（只删这个命名模式，不动目录里的其他文件）。
    """

    holdout_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for image_path, index in frames:
        name = f"frame_{index:04d}{image_path.suffix}"
        _link(image_path, holdout_dir / name)
        digests[name] = _file_sha256(holdout_dir / name)
    for stale in sorted(holdout_dir.glob("frame_*.png")):
        if stale.name not in digests:
            stale.unlink()
    manifest = holdout_dir / "frames.sha256"
    manifest.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())),
        encoding="utf-8",
    )
    return {
        "dir": str(holdout_dir).replace("\\", "/"),
        "frames": len(digests),
        "manifest": str(manifest).replace("\\", "/"),
    }


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
    holdout_blocks: Sequence[int] = DEFAULT_HOLDOUT_BLOCKS,
    holdout_dir: Path | None = None,
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
        "holdout_blocks": list(holdout_blocks),
        "splits": {},
        "ignored_boxes_skipped": 0,
        "boxes": {"train": 0, "val": 0, "holdout": 0},
        "fixtures": None,
    }

    for split in ("train", "val", "holdout"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        stats["splits"][split] = {"frames": 0, "empty_frames": 0}

    holdout_frames: list[tuple[Path, int]] = []
    for items in grouped.values():
        train_idx, val_idx, holdout_idx = split_block_indices(
            len(items), blocks=blocks, val_blocks=val_blocks, holdout_blocks=holdout_blocks
        )
        for split, indices in (
            ("train", train_idx),
            ("val", val_idx),
            ("holdout", holdout_idx),
        ):
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
                if split == "holdout":
                    holdout_frames.append((image_path, int(record.get("index", position))))

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
    if holdout_frames:
        holdout_yaml = out_dir / "data_holdout.yaml"
        holdout_yaml.write_text(
            "\n".join(
                [
                    f"path: {out_dir.resolve().as_posix()}",
                    "train: images/train",
                    "val: images/holdout",
                    f"nc: {len(CLASS_NAMES)}",
                    f"names: {CLASS_NAMES}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        stats["data_yaml_holdout"] = str(holdout_yaml).replace("\\", "/")
    if holdout_dir is not None and holdout_frames:
        stats["fixtures"] = write_holdout_fixtures(holdout_dir, holdout_frames)
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _parse_blocks(text: str) -> tuple[int, ...]:
    """解析 `--val-blocks` / `--holdout-blocks` 的「逗号分隔块号」写法。"""

    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="annotation JSON → YOLO 数据集")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/phase2/yolo_dataset"))
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--val-blocks",
        default=",".join(str(item) for item in DEFAULT_VAL_BLOCKS),
        help="用哪些连续块做验证集（早停用；逗号分隔）",
    )
    parser.add_argument(
        "--holdout-blocks",
        default=",".join(str(item) for item in DEFAULT_HOLDOUT_BLOCKS),
        help="用哪些连续块做 holdout（永不参与训练与早停；留空字符串表示不要 holdout）",
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=DEFAULT_HOLDOUT_DIR,
        help="holdout 帧与 frames.sha256 的落盘目录（传 '-' 表示只写数据集、不落 fixtures）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    holdout_dir = None if str(args.holdout_dir) == "-" else args.holdout_dir
    stats = build(
        args.annotation,
        args.out,
        blocks=args.blocks,
        val_blocks=_parse_blocks(args.val_blocks),
        holdout_blocks=_parse_blocks(args.holdout_blocks),
        holdout_dir=holdout_dir,
    )
    splits = stats["splits"]
    print(
        f"数据集已生成：train {splits['train']['frames']} 帧"
        f"（{stats['boxes']['train']} 框，其中空帧 {splits['train']['empty_frames']}）｜"
        f"val {splits['val']['frames']} 帧（{stats['boxes']['val']} 框）｜"
        f"holdout {splits['holdout']['frames']} 帧（{stats['boxes']['holdout']} 框）"
    )
    print(f"跳过 ignored 框 {stats['ignored_boxes_skipped']} 个｜data.yaml: {stats['data_yaml']}")
    if stats["fixtures"] is not None:
        fixtures = stats["fixtures"]
        print(f"holdout fixtures：{fixtures['frames']} 帧 → {fixtures['manifest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

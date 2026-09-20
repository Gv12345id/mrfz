"""从战斗帧里自动提取数字模板，供 OCR 使用。

游戏字体的位图属于游戏素材，按 AGENTS.md §7 不入库；所以仓库里只放这个提取工具，
模板落到 `.gitignore` 挡住的 `assets/templates/digit_templates.npz`。

**两个区域、两套模板**（`--source`）：右下角费用区 `cost`、底部干员卡区 `cards`。
两处字号与抗锯齿不同，同一个数字在两处的海明距离实测可达 292 位（阈值只有 120），
所以各聚各的类、各写各的模板，绝不能合并成一套均值模板。

流程：

1. 在对应区域抠出白色数字连通域（高 20–46px、宽 6–34px、长宽比 0.25–1.0）；
2. 归一化到 24×32 的二值图，用海明距离贪心聚类；
3. 打印/导出聚类蒙太奇，人工给出"簇 → 数字"的映射；
4. 按 0–9 顺序平均成 10 张模板，写进 npz
   （`cost` → `templates`，`cards` → `card_templates`）。

用法::

    # 第一步：先看聚类蒙太奇，确认每个簇是哪个数字（两个区域都跑）
    .\\.venv\\Scripts\\python.exe tools\\build_digit_templates.py \\
        --annotation runs/phase2/annotation_400.json --source both

    # 第二步：按蒙太奇里的簇号补上标签，生成模板
    .\\.venv\\Scripts\\python.exe tools\\build_digit_templates.py \\
        --annotation runs/phase2/annotation_400.json --source both \\
        --labels "7,6,1,8,5,0,2,4,3,9,0" --card-labels "3,1,6,8,0,5,2,9,4,7"

只跑 `--source cards` 时会保留 npz 里已有的费用模板（就地合并，不会把它冲掉）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from arknights_agent.perception.ocr import extract_card_glyphs, extract_glyphs

CLUSTER_DISTANCE = 90  # 24×32=768 位里允许多少位不同
SOURCES = ("cost", "cards")
DEFAULT_OUT = Path("assets/templates/digit_templates.npz")
DEFAULT_MONTAGE = Path("runs/phase2/preview/digit_clusters.png")
DEFAULT_CARD_MONTAGE = Path("runs/phase2/preview/card_clusters.png")

Mask = NDArray[np.bool_]


def collect_glyphs(records: Sequence[Mapping[str, Any]], source: str) -> list[Mask]:
    """从帧里抠出指定区域的字形：`cost` = 右下费用区，`cards` = 底部干员卡区。

    字形抠取复用 `arknights_agent.perception.ocr` 的实现——工具与推理必须用同一套
    ROI、阈值与归一化尺寸，否则模板和现场字形对不上。
    """

    extractor = extract_glyphs if source == "cost" else extract_card_glyphs
    glyphs: list[Mask] = []
    for record in records:
        image = cv2.imread(str(record["screenshot"]))
        if image is None:
            continue
        glyphs.extend(extractor(image))
    return glyphs


def parse_labels(text: str | None) -> list[int] | None:
    """解析 `--labels` / `--card-labels` 的逗号分隔簇标签。"""

    if text is None:
        return None
    return [int(item) for item in text.split(",") if item.strip()]


def cluster_glyphs(glyphs: Sequence[Mask]) -> list[dict[str, object]]:
    """海明距离贪心聚类，簇按样本数降序返回。"""

    clusters: list[dict[str, object]] = []
    for glyph in glyphs:
        for cluster in clusters:
            center = cluster["center"]
            assert isinstance(center, np.ndarray)
            if int(np.count_nonzero(glyph != center)) < CLUSTER_DISTANCE:
                members = cluster["members"]
                assert isinstance(members, list)
                members.append(glyph)
                cluster["center"] = np.mean(np.stack(members), axis=0) > 0.5
                break
        else:
            clusters.append({"center": glyph, "members": [glyph]})
    clusters.sort(key=lambda item: -len(item["members"]))  # type: ignore[arg-type]
    return clusters


def render_montage(clusters: Sequence[dict[str, object]], path: Path) -> None:
    """把各簇中心拼成一张图，便于人工确认每个簇是哪个数字。"""

    tiles = []
    for index, cluster in enumerate(clusters):
        center = cluster["center"]
        assert isinstance(center, np.ndarray)
        tile = np.where(center, 255, 0).astype(np.uint8)
        tile = cv2.resize(tile, (48, 64), interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(tile, 24, 6, 6, 6, cv2.BORDER_CONSTANT, value=60)
        cv2.putText(tile, str(index), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        tiles.append(tile)
    if not tiles:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(tiles))


def build_templates(
    clusters: Sequence[dict[str, object]], labels: Sequence[int]
) -> NDArray[np.bool_]:
    """按数字标签把簇平均成 0–9 的模板（每个数字形状取均值）。"""

    if len(labels) != len(clusters):
        raise SystemExit(f"--labels 数量（{len(labels)}）必须与聚类数（{len(clusters)}）一致")
    groups: dict[int, list[Mask]] = {digit: [] for digit in range(10)}
    for cluster, digit in zip(clusters, labels, strict=True):
        if not 0 <= digit <= 9:
            raise SystemExit(f"标签必须是 0–9，收到 {digit}")
        center = cluster["center"]
        assert isinstance(center, np.ndarray)
        groups[digit].append(center)
    templates = []
    for digit in range(10):
        if not groups[digit]:
            raise SystemExit(f"数字 {digit} 没有任何簇，模板不完整")
        templates.append(np.mean(np.stack(groups[digit]), axis=0) > 0.5)
    return np.stack(templates)


def write_templates(
    path: Path,
    built: Mapping[str, NDArray[np.bool_]],
    audit: Mapping[str, dict[str, Any]],
    *,
    frames: int,
) -> None:
    """写 npz：`cost` → `templates`，`cards` → `card_templates`。

    只写卡片模板时，先把已有 npz 里的费用模板读出来一起写回去——npz 是整文件覆盖，
    不合并就会把费用模板冲掉。
    """

    payload: dict[str, Any] = {}
    if "cost" not in built and path.is_file():
        with np.load(path) as existing:
            if "templates" in existing:
                payload["templates"] = np.asarray(existing["templates"], dtype=bool)
    if "cost" in built:
        payload["templates"] = built["cost"]
    if "cards" in built:
        payload["card_templates"] = built["cards"]
    if "templates" not in payload:
        raise SystemExit(f"{path} 里没有费用模板：先用 --source cost 或 --source both 生成一次")
    payload["source_frames"] = np.array(frames, dtype=np.int32)
    for source, entry in audit.items():
        prefix = "" if source == "cost" else "cards_"
        payload[f"{prefix}cluster_labels"] = np.array(entry["labels"], dtype=np.int16)
        payload[f"{prefix}cluster_sizes"] = np.array(entry["sizes"], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从战斗帧提取数字模板")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200, help="用多少帧做提取")
    parser.add_argument(
        "--source",
        choices=("cost", "cards", "both"),
        default="cost",
        help="抠哪个区域的字形：cost=右下费用区，cards=底部干员卡，both=两个都做",
    )
    parser.add_argument("--labels", default=None, help="费用区的簇→数字映射，如 7,6,1,8,5,0,...")
    parser.add_argument("--card-labels", default=None, help="卡片区的簇→数字映射（同上格式）")
    parser.add_argument("--montage", type=Path, default=DEFAULT_MONTAGE)
    parser.add_argument("--card-montage", type=Path, default=DEFAULT_CARD_MONTAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    records = [item for item in data["records"] if item.get("result") == "ok"][: args.frames]
    wanted = SOURCES if args.source == "both" else (args.source,)
    label_args = {"cost": args.labels, "cards": args.card_labels}
    montages = {"cost": args.montage, "cards": args.card_montage}
    flag_names = {"cost": "--labels", "cards": "--card-labels"}

    built: dict[str, NDArray[np.bool_]] = {}
    audit: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for source in wanted:
        glyphs = collect_glyphs(records, source)
        clusters = cluster_glyphs(glyphs)
        sizes = [len(cluster["members"]) for cluster in clusters]  # type: ignore[arg-type]
        render_montage(clusters, montages[source])
        print(
            f"[{source}] 从 {len(records)} 帧抠出 {len(glyphs)} 个数字块，聚成 {len(clusters)} 簇"
        )
        print(f"[{source}] 簇样本数：{sizes}")
        print(f"[{source}] 蒙太奇：{montages[source]}")
        labels = parse_labels(label_args[source])
        if labels is None:
            missing.append(source)
            continue
        built[source] = build_templates(clusters, labels)
        audit[source] = {"labels": labels, "sizes": sizes}

    if missing:
        for source in missing:
            example = "7,6,1,8,5,0,2,4,3,9,0" if source == "cost" else "3,1,6,8,0,5,2,9,4,7"
            print(
                f"请先看蒙太奇，再用 {flag_names[source]} 传 {source} 区的簇→数字映射"
                f"（簇号顺序，例如 {flag_names[source]} {example}）"
            )
        return 0

    write_templates(args.out, built, audit, frames=len(records))
    shapes = {source: template.shape for source, template in built.items()}
    print(f"模板已保存：{args.out}（形状 {shapes}，按 0–9 顺序）")
    for source, entry in audit.items():
        print(f"审计[{source}]：簇标签={entry['labels']}｜簇样本数={entry['sizes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

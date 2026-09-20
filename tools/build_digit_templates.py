"""从战斗帧里自动提取数字模板，供 OCR 使用。

游戏字体的位图属于游戏素材，按 AGENTS.md §7 不入库；所以仓库里只放这个提取工具，
模板落到 `.gitignore` 挡住的 `assets/templates/digit_templates.npz`。

流程：

1. 在费用 ROI 里抠出白色数字连通域（高 20–46px、宽 6–34px、长宽比 0.25–1.0）；
2. 归一化到 24×32 的二值图，用海明距离贪心聚类；
3. 打印/导出聚类蒙太奇，人工（或 `--labels`）给出"簇 → 数字"的映射；
4. 按 0–9 顺序平均成 10 张模板，写进 npz。

用法::

    # 第一步：先看聚类蒙太奇，确认每个簇是哪个数字
    .\\.venv\\Scripts\\python.exe tools\\build_digit_templates.py ^
        --annotation runs/phase2/annotation_400.json

    # 第二步：把簇号对应的数字按顺序传进来，生成模板
    .\\.venv\\Scripts\\python.exe tools\\build_digit_templates.py ^
        --annotation runs/phase2/annotation_400.json --labels "7,6,1,8,5,0,2,4,3,9,0"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# 费用数字所在的 ROI（x, y, w, h），1280×720 逻辑坐标系。
COST_ROI: tuple[int, int, int, int] = (1200, 487, 80, 50)
GLYPH_SIZE = (24, 32)  # (w, h)
CLUSTER_DISTANCE = 90  # 24×32=768 位里允许多少位不同
MIN_GLYPH_HEIGHT = 20
MAX_GLYPH_HEIGHT = 46
MIN_GLYPH_WIDTH = 6
MAX_GLYPH_WIDTH = 34
MIN_ASPECT = 0.25
MAX_ASPECT = 1.0
DEFAULT_OUT = Path("assets/templates/digit_templates.npz")
DEFAULT_MONTAGE = Path("runs/phase2/preview/digit_clusters.png")

Mask = NDArray[np.bool_]


def extract_glyphs(
    image: NDArray[np.uint8], roi: tuple[int, int, int, int] = COST_ROI
) -> list[Mask]:
    """抠出一帧费用区里的数字块（左→右排序）。"""

    x, y, w, h = roi
    patch = image[y : y + h, x : x + w]
    if patch.size == 0:
        return []
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 80)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[int, Mask]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if not (MIN_GLYPH_HEIGHT <= bh <= MAX_GLYPH_HEIGHT):
            continue
        if not (MIN_GLYPH_WIDTH <= bw <= MAX_GLYPH_WIDTH):
            continue
        if not (MIN_ASPECT <= bw / bh <= MAX_ASPECT):
            continue
        glyph = mask[by : by + bh, bx : bx + bw]
        found.append((bx, cv2.resize(glyph, GLYPH_SIZE, interpolation=cv2.INTER_AREA) > 127))
    found.sort(key=lambda item: item[0])
    return [glyph for _, glyph in found]


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从战斗帧提取数字模板")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200, help="用多少帧做提取")
    parser.add_argument("--labels", default=None, help="逗号分隔的簇→数字映射，如 7,6,1,8,5,0,...")
    parser.add_argument("--montage", type=Path, default=DEFAULT_MONTAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    records = [item for item in data["records"] if item.get("result") == "ok"][: args.frames]
    glyphs: list[Mask] = []
    for record in records:
        image = cv2.imread(str(record["screenshot"]))
        if image is None:
            continue
        glyphs.extend(extract_glyphs(image))
    clusters = cluster_glyphs(glyphs)
    render_montage(clusters, args.montage)
    print(f"从 {len(records)} 帧抠出 {len(glyphs)} 个数字块，聚成 {len(clusters)} 簇")
    print(f"簇样本数：{[len(c['members']) for c in clusters]}")  # type: ignore[arg-type]
    print(f"蒙太奇：{args.montage}")
    if args.labels is None:
        print(
            "请查看蒙太奇后传入 --labels（按簇号顺序写数字），例如 --labels 7,6,1,8,5,0,2,4,3,9,0"
        )
        return 0
    labels = [int(item) for item in args.labels.split(",") if item.strip()]
    templates = build_templates(clusters, labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        templates=templates,
        # 审计信息：簇顺序、各簇样本数、用了哪些帧。簇顺序变了但标签没跟着变，
        # 是这套流程最容易踩的坑（会静默把 6 认成 7），所以把来源一并存下来。
        cluster_labels=np.array(labels, dtype=np.int16),
        cluster_sizes=np.array([len(c["members"]) for c in clusters], dtype=np.int32),  # type: ignore[arg-type]
        source_frames=np.array(len(records), dtype=np.int32),
    )
    print(f"模板已保存：{args.out}（形状 {templates.shape}，按 0–9 顺序）")
    print(f"审计：簇标签={labels}｜簇样本数={[len(c['members']) for c in clusters]}")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    sys.exit(main())

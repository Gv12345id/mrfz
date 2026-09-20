"""阶段 2 标注工具：自动预标注 + 人工抽查修正（tkinter）。

工作方式（对应 PLANS.md 阶段 2 任务 1）：

1. **自动预标注**：对每帧先生成初始框，人只需要改错，不用从零框选。
   - `--pre-annotate heuristic`：无需任何权重。用相邻帧差分找"在动的东西"，
     再按尺寸 / 长宽比 / UI 屏蔽区过滤，得到敌人候选框。
   - `--pre-annotate yolo --weights models/xxx.pt`：用训好的 YOLOv8 权重跑一遍
     （阶段 3 产出权重后即可切换到这条路）。
2. **人工抽查**：GUI 里逐帧看，框错了就删/重画，按 S 保存。

快捷键：

===== ==================== ==========================================
按键   动作                 说明
===== ==================== ==========================================
← / A 上一帧               上一帧
→ / D 下一帧               下一帧
P     上一张待检查帧       跳到上一张 `reviewed=false` 的帧
N     下一张待检查帧       跳到下一张 `reviewed=false` 的帧
X     删除选中框           删掉当前选中的框（也可按 Delete）
E     标记忽略             把选中的框标成"非敌人/忽略"（灰色虚线保留，训练时排除）
T     批量标记忽略         把相邻 ±3 帧里同一位置的自动框一起标为忽略（特效常连续出现）
R     重新框选             进入画框模式，按住左键拖出新框；再按 R 退出
S     保存                 把 labels 与 reviewed 写回 annotation JSON
空格  标记已检查           切换当前帧的 reviewed 状态
Q     退出                 未保存时会提示
===== ==================== ==========================================

用法::

    # 先跑自动预标注（启发式，无需权重）
    .\\.venv\\Scripts\\python.exe tools\\label.py --annotation runs/phase2/annotation_battle.json \\
        --pre-annotate heuristic

    # 再开 GUI 抽查
    .\\.venv\\Scripts\\python.exe tools\\label.py --annotation runs/phase2/annotation_battle.json

    # 有 YOLO 权重后
    .\\.venv\\Scripts\\python.exe tools\\label.py --annotation runs/phase2/annotation_battle.json \\
        --pre-annotate yolo --weights models/yolov8n_arknights.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from arknights_agent.imaging import read_bgr

# 敌人候选框的尺寸范围（1280×720 坐标），落在范围外的连通域直接丢。
MIN_BOX_SIDE = 18
MAX_BOX_SIDE = 140
MIN_BOX_AREA = 260
MAX_BOX_AREA = 14000

# UI 屏蔽区（x1, y1, x2, y2）：顶部 HUD、底部干员栏、右上控制、右下费用/提示。
UI_MASKS: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 1280, 62),
    (0, 570, 1280, 720),
    (1020, 0, 1280, 96),
    (1070, 470, 1280, 570),
)

# 单帧预标注上限：0-1 关卡同时在场的敌人远少于这个数，超出的基本都是噪声。
MAX_LABELS = 6

# 全局运动（镜头平移）处理：
# - PAN_SHIFT_PX：相位相关估计出的平移超过它，就认为镜头在移动，先做运动补偿再差分；
# - RESIDUAL_MOTION_RATIO：补偿后画面仍有这么大比例在变，判定为整屏变化，该帧留空；
# - PAN_ROI：估计平移用的区域，避开顶部 HUD 与底部干员栏。
PAN_SHIFT_PX = 6.0
# 相位相关的可信度下限与平移幅度上限：合成背景/UI 覆盖时估计值会离谱，
# 这时宁可当作"没有平移"，也不要用错误的对齐把真正的目标抹掉。
PAN_MIN_RESPONSE = 0.10
PAN_MAX_PX = 80.0
# 相位相关需要纹理：ROI 过于平坦（标准差低于它）时估计值不可信，直接当作没平移。
PAN_MIN_TEXTURE = 8.0
# 补偿后需要抹掉的边缘带（像素）：平移补偿在边界处必然产生伪影。
PAN_EDGE_MARGIN = 8
RESIDUAL_MOTION_RATIO = 0.25
# 两块互不相交的背景取样区：只有它们估出的平移一致，才认定"镜头在动"。
# 单块 ROI 会把"唯一动目标的位移"误当成镜头平移，从而把目标补偿掉。
PAN_REGIONS: tuple[tuple[int, int, int, int], ...] = (
    (140, 80, 640, 360),
    (660, 300, 1140, 560),
)
# 两块 ROI 估计出的平移差超过它，就认为估计不可信，按"镜头没动"处理。
PAN_CONSISTENCY_PX = 3.0

# 合并判据：IoU、包含率（交集/较小框面积）任一超阈值，或中心距小于短边的这个比例，
# 都视为"同一个目标"，合并成外接矩形。
IOU_MERGE_THRESHOLD = 0.15
CONTAINMENT_MERGE_THRESHOLD = 0.35
CENTER_MERGE_FACTOR = 0.60
# 只有补偿能把残差压到基线的这个比例以下，才认为"镜头确实在平移"。
PAN_ACCEPT_RESIDUAL_RATIO = 0.80

# 时间一致性过滤：至少有一个相邻帧里存在中心落在半径内的框，才认为它是稳定目标
# （真敌人）；只在单帧闪一下的弹道 / 火花会被丢掉。邻居只在同一批内比较。
TEMPORAL_RADIUS_PX = 28.0
TEMPORAL_MIN_HITS = 1
# 按 T 批量标记忽略时，向两侧各覆盖多少帧。
NEIGHBOUR_MARK_FRAMES = 3

# 发光/粒子特效过滤：部署蓝光、爆炸火光、技能粒子这类区域里，高亮 + 高饱和像素占比
# 很高；敌人立绘有深色描边与纹理，占比明显更低。
GLOW_VALUE_MIN = 200
GLOW_SAT_MIN = 100
GLOW_FRACTION_MAX = 0.45

# 静态场景装饰过滤：地面红三角/光圈这类装饰在同一位置连续多帧像素几乎不变；
# 敌人即使站着不动也有待机/受击动画。连续这么多帧几乎不动就判为装饰。
STATIC_DIFF_THRESHOLD = 2.0
STATIC_MIN_FRAMES = 3

ENEMY_CLASS_ID = 0
ENEMY_CLASS_NAME = "enemy"

KEY_ACTIONS: dict[str, str] = {
    "Left": "prev_frame",
    "a": "prev_frame",
    "A": "prev_frame",
    "Right": "next_frame",
    "d": "next_frame",
    "D": "next_frame",
    "p": "prev_unreviewed",
    "P": "prev_unreviewed",
    "n": "next_unreviewed",
    "N": "next_unreviewed",
    "x": "delete_box",
    "X": "delete_box",
    "Delete": "delete_box",
    "e": "toggle_ignored",
    "E": "toggle_ignored",
    "t": "toggle_ignored_neighbours",
    "T": "toggle_ignored_neighbours",
    "r": "toggle_draw",
    "R": "toggle_draw",
    "s": "save",
    "S": "save",
    "space": "toggle_reviewed",
    "q": "quit",
    "Q": "quit",
}


def key_action(keysym: str) -> str | None:
    """把 tkinter 的按键名映射成动作名（GUI 之外可单测）。"""

    return KEY_ACTIONS.get(keysym)


def make_label(box: Sequence[float], *, source: str, score: float | None = None) -> dict[str, Any]:
    """构造一个标注框：像素坐标 xyxy + 类别 + 来源（auto/human）。"""

    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((max(0.0, x1), min(1279.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(719.0, y2)))
    label: dict[str, Any] = {
        "class_id": ENEMY_CLASS_ID,
        "class_name": ENEMY_CLASS_NAME,
        "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "source": source,
    }
    if score is not None:
        label["score"] = round(float(score), 3)
    return label


def label_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """两个标注框的 IoU，用于合并重叠候选。"""

    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def label_containment(a: dict[str, Any], b: dict[str, Any]) -> float:
    """包含率 = 交集 / 较小框面积，用来识别"大框套小框"这类低 IoU 重叠。"""

    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return 0.0 if smaller <= 0 else inter / smaller


def _center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    dx = (ax1 + ax2) / 2 - (bx1 + bx2) / 2
    dy = (ay1 + ay2) / 2 - (by1 + by2) / 2
    return float((dx * dx + dy * dy) ** 0.5)


def _short_side(label: dict[str, Any]) -> float:
    x1, y1, x2, y2 = label["xyxy"]
    return min(x2 - x1, y2 - y1)


def should_merge(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """两个候选框是否属于同一个目标。"""

    if label_iou(a, b) > IOU_MERGE_THRESHOLD:
        return True
    if label_containment(a, b) > CONTAINMENT_MERGE_THRESHOLD:
        return True
    threshold = CENTER_MERGE_FACTOR * min(_short_side(a), _short_side(b))
    return _center_distance(a, b) < threshold


def _union_labels(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """两个框并成外接矩形；来源按"人工优先"保留，置信度取最大。"""

    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    source = "human" if "human" in (a.get("source"), b.get("source")) else a.get("source")
    merged = make_label([min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)], source=source)
    scores = [item["score"] for item in (a, b) if "score" in item]
    if scores:
        merged["score"] = round(max(scores), 3)
    return merged


def _overlaps_ignored(box: dict[str, Any], region: dict[str, Any]) -> bool:
    """候选框是否落在"人工标记忽略"的区域里（按 IoU 判）。"""

    region_box = make_label(region["xyxy"], source="human")
    return label_iou(box, region_box) > 0.3


def merge_labels(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同一个目标的多个候选框聚成一个，并按位置稳定排序。

    只靠 IoU 去重不够：镜头平移或敌人快速移动时，同一个敌人会留下"影子框"，
    两框可能几乎不重叠（IoU 很低）却中心很近，或者一个大框套住一个小框。
    这里用 IoU / 包含率 / 中心距三条一起判，判为同一个就取外接矩形。
    """

    ordered = sorted(labels, key=_label_area, reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        for index, existing in enumerate(kept):
            if should_merge(candidate, existing):
                kept[index] = _union_labels(candidate, existing)
                break
        else:
            kept.append(candidate)
    # 稳定排序：先上后下、先左后右，编号连续可预期。
    kept.sort(key=lambda item: (item["xyxy"][1], item["xyxy"][0]))
    return kept[:MAX_LABELS]


def _label_area(label: dict[str, Any]) -> float:
    x1, y1, x2, y2 = label["xyxy"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _in_ui_mask(cx: float, cy: float) -> bool:
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in UI_MASKS)


def glow_fraction(frame: NDArray[np.uint8], box: Sequence[float]) -> float:
    """框内"高亮 + 高饱和"像素占比，用来识别发光/粒子特效。"""

    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    patch = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if patch.size == 0:
        return 1.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] > GLOW_VALUE_MIN) & (hsv[:, :, 1] > GLOW_SAT_MIN)
    return float(bright.mean())


def is_glow_like(frame: NDArray[np.uint8], box: Sequence[float]) -> bool:
    """是不是发光/粒子特效（部署蓝光、爆炸火光、技能粒子）。"""

    return glow_fraction(frame, box) > GLOW_FRACTION_MAX


def estimate_global_shift(
    prev_gray: NDArray[np.uint8], cur_gray: NDArray[np.uint8]
) -> tuple[float, float]:
    """估计两帧之间的整体平移量（像素）；不可信时返回 (0, 0)。

    在两块互不相交的背景区各估一次，只有两处结论一致（差值小于 PAN_CONSISTENCY_PX）
    才认为镜头真的在平移。单块 ROI 的估计会被"画面里唯一在动的那个目标"带偏 ——
    那种情况下补偿会把真正要检测的目标一起抹掉。
    """

    estimates: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in PAN_REGIONS:
        first = prev_gray[y1:y2, x1:x2]
        second = cur_gray[y1:y2, x1:x2]
        if float(first.std()) < PAN_MIN_TEXTURE or float(second.std()) < PAN_MIN_TEXTURE:
            continue
        (dx, dy), response = cv2.phaseCorrelate(np.float32(first), np.float32(second))
        if response < PAN_MIN_RESPONSE or abs(dx) > PAN_MAX_PX or abs(dy) > PAN_MAX_PX:
            continue
        estimates.append((float(dx), float(dy)))
    if len(estimates) < 2:
        return 0.0, 0.0
    (ax, ay), (bx, by) = estimates[0], estimates[1]
    if abs(ax - bx) > PAN_CONSISTENCY_PX or abs(ay - by) > PAN_CONSISTENCY_PX:
        return 0.0, 0.0
    return (ax + bx) / 2, (ay + by) / 2


def align_frame(
    prev_gray: NDArray[np.uint8], cur_gray: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], tuple[float, float]]:
    """把上一帧对齐到当前帧，返回（对齐后的上一帧, 实际使用的平移量）。

    镜头在平移时，直接差分会把整片背景当移动目标。这里先估平移量，再把上一帧平移
    回去；`phaseCorrelate` 的符号约定容易记反，所以两个方向都试一遍。**只有补偿把
    残差压到基线的 0.8 倍以下才采纳**：否则（例如画面近乎纯色、估计值离谱时）保持
    不对齐，免得把真正在动的目标一起补偿掉。
    """

    baseline = float(np.mean(cv2.absdiff(prev_gray, cur_gray)))
    dx, dy = estimate_global_shift(prev_gray, cur_gray)
    if (dx * dx + dy * dy) ** 0.5 <= PAN_SHIFT_PX:
        return prev_gray, (0.0, 0.0)
    height, width = prev_gray.shape[:2]
    best: tuple[NDArray[np.uint8], tuple[float, float], float] = (
        prev_gray,
        (0.0, 0.0),
        baseline,
    )
    for sign in (1.0, -1.0):
        matrix = np.float32([[1, 0, sign * dx], [0, 1, sign * dy]])
        warped = cv2.warpAffine(prev_gray, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        residual = float(np.mean(cv2.absdiff(warped, cur_gray)))
        if residual < best[2]:
            best = (warped, (sign * dx, sign * dy), residual)
    if best[2] >= baseline * PAN_ACCEPT_RESIDUAL_RATIO:
        return prev_gray, (0.0, 0.0)
    return best[0], best[1]


def heuristic_labels(
    prev_frame: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    *,
    pixel_threshold: int = 18,
) -> list[dict[str, Any]]:
    """用帧间差分找移动目标，作为敌人候选框（无权重时的预标注）。

    先估计全局平移（镜头移动）。镜头在动时，整片背景都在"动"，直接差分出来的框几乎
    全是噪声 —— 所以先用相位相关估出平移量，把上一帧对齐回来再差分，只留下"自己会
    动"的目标；对齐后画面仍有大比例变化（真正的整屏切换）则留空交给人工。
    """

    if prev_frame.shape != frame.shape:
        return []
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aligned, shift = align_frame(gray_prev, gray_cur)
    delta = cv2.absdiff(aligned, gray_cur)
    # 平移补偿后边缘像素不可信，抹掉一圈，避免边缘伪影变成框。
    margin = int(abs(shift[0])) + int(abs(shift[1])) + PAN_EDGE_MARGIN
    delta[:margin, :] = 0
    delta[-margin:, :] = 0
    delta[:, :margin] = 0
    delta[:, -margin:] = 0
    moved_ratio = float(np.count_nonzero(delta > pixel_threshold)) / delta.size
    if moved_ratio > RESIDUAL_MOTION_RATIO:
        return []
    _, binary = cv2.threshold(delta, pixel_threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not (MIN_BOX_SIDE <= w <= MAX_BOX_SIDE and MIN_BOX_SIDE <= h <= MAX_BOX_SIDE):
            continue
        if not (MIN_BOX_AREA <= area <= MAX_BOX_AREA):
            continue
        if _in_ui_mask(x + w / 2, y + h / 2):
            continue
        if is_glow_like(frame, (x, y, x + w, y + h)):
            continue  # 发光/粒子特效：部署蓝光、爆炸火光、技能粒子
        labels.append(make_label([x, y, x + w, y + h], source="auto"))
    return merge_labels(labels)


def temporal_filter(
    data: dict[str, Any],
    *,
    radius: float = TEMPORAL_RADIUS_PX,
    min_hits: int = TEMPORAL_MIN_HITS,
) -> dict[str, int]:
    """丢掉只在单帧出现的自动框。

    战斗里"会动"的不只是敌人：弹道、技能特效、火花粒子都会动，但它们通常只出现
    一两帧；敌人则会连续多帧稳定出现在同一位置附近。判据是"相邻两帧里各有一个框
    落在这个框中心 28px 内"。人工框不受影响。
    """

    records = [item for item in data["records"] if item.get("result") == "ok"]
    stats = {"kept": 0, "removed": 0, "frames_touched": 0}
    for position, record in enumerate(records):
        boxes = record.get("labels") or []
        neighbours: list[dict[str, Any]] = []
        for offset in (-1, 1):
            other_index = position + offset
            if not (0 <= other_index < len(records)):
                continue
            other = records[other_index]
            if record.get("batch") != other.get("batch"):
                continue  # 不同批次之间时间不连续，不做比较
            neighbours.extend(other.get("labels") or [])
        kept: list[dict[str, Any]] = []
        for box in boxes:
            if box.get("source") == "human":
                kept.append(box)
                continue
            hits = sum(1 for other in neighbours if _center_distance(box, other) <= radius)
            if hits >= min_hits:
                kept.append(box)
            else:
                stats["removed"] += 1
        if len(kept) != len(boxes):
            stats["frames_touched"] += 1
        record["labels"] = kept
        stats["kept"] += len(kept)
    return stats


def static_filter(
    data: dict[str, Any],
    *,
    threshold: float = STATIC_DIFF_THRESHOLD,
    min_frames: int = STATIC_MIN_FRAMES,
) -> dict[str, int]:
    """丢掉"像素几乎不变"的静态装饰框（地面红三角、光圈等）。

    敌人即使站定不动，也有待机 / 受击动画，同一位置的像素会持续变化；场景装饰则是
    逐帧完全一致。判据：同一坐标区域连续 `min_frames` 帧的平均灰度差都低于阈值。
    """

    records = [item for item in data["records"] if item.get("result") == "ok"]
    stats = {"removed": 0, "frames_touched": 0}
    for position, record in enumerate(records):
        boxes = record.get("labels") or []
        kept: list[dict[str, Any]] = []
        touched = False
        for box in boxes:
            if box.get("source") == "human":
                kept.append(box)
                continue
            if _is_static(data, records, position, box, threshold=threshold, min_frames=min_frames):
                stats["removed"] += 1
                touched = True
                continue
            kept.append(box)
        record["labels"] = kept
        if touched:
            stats["frames_touched"] += 1
    return stats


def _is_static(
    data: dict[str, Any],
    records: list[dict[str, Any]],
    position: int,
    box: dict[str, Any],
    *,
    threshold: float,
    min_frames: int,
) -> bool:
    """看同一个坐标区域在前后几帧里是否几乎不变。

    只往后看的话，序列末尾几帧永远判不出来；所以先往后数，不够再往前数。
    """

    forward = _stable_run(records, position, box, threshold=threshold, direction=1)
    if forward >= min_frames:
        return True
    backward = _stable_run(records, position, box, threshold=threshold, direction=-1)
    # 序列中间/末尾的帧前后都凑不满 min_frames，把两个方向合起来数。
    return forward + backward >= min_frames


def _stable_run(
    records: list[dict[str, Any]],
    position: int,
    box: dict[str, Any],
    *,
    threshold: float,
    direction: int,
) -> int:
    """沿着 direction 方向数"与本帧几乎一致"的连续帧数。"""

    x1, y1, x2, y2 = (int(round(float(v))) for v in box["xyxy"])
    reference = cv2.imread(str(records[position]["screenshot"]), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        return 0
    stable = 0
    for offset in range(1, STATIC_MIN_FRAMES + 1):
        other_index = position + direction * offset
        if not (0 <= other_index < len(records)):
            break
        if records[other_index].get("batch") != records[position].get("batch"):
            break
        other = cv2.imread(str(records[other_index]["screenshot"]), cv2.IMREAD_GRAYSCALE)
        if other is None or other.shape != reference.shape:
            break
        first = reference[y1:y2, x1:x2]
        second = other[y1:y2, x1:x2]
        if first.size == 0 or second.size == 0:
            break
        if float(np.mean(cv2.absdiff(first, second))) < threshold:
            stable += 1
        else:
            break
    return stable


def yolo_labels(
    weights: Path, frame: NDArray[np.uint8], *, conf: float = 0.25
) -> list[dict[str, Any]]:
    """用 Ultralytics YOLO 权重跑一遍（需要先训练出权重）。"""

    from ultralytics import YOLO  # 延迟导入：不跑 YOLO 时不需要加载 torch

    model = YOLO(str(weights))
    results = model.predict(frame, conf=conf, verbose=False)
    labels: list[dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=False):
            labels.append(make_label(xyxy, source="auto", score=score))
    return merge_labels(labels)


def load_annotation(path: Path) -> dict[str, Any]:
    """读取 annotation JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _detect(
    mode: str,
    previous_frame: NDArray[np.uint8] | None,
    frame: NDArray[np.uint8],
    weights: Path | None,
    conf: float,
) -> list[dict[str, Any]]:
    """按模式产出候选框。"""

    if mode == "heuristic":
        return [] if previous_frame is None else heuristic_labels(previous_frame, frame)
    if mode == "yolo":
        if weights is None:
            raise SystemExit("--pre-annotate yolo 需要同时给 --weights 指定权重文件")
        return yolo_labels(weights, frame, conf=conf)
    raise SystemExit(f"未知的预标注模式：{mode}")


def renumber_frames(data: dict[str, Any], target_dir: Path) -> int:
    """把帧整理成连续编号（硬链接，零拷贝），并回写 `screenshot`。

    不同批次混在一起时原始文件名会跳号；这里统一成 `frame_0001.png …`，原始文件名
    记在 `source_frame` 里备查。函数是幂等的：第二次调用会从 `original_screenshot`
    重新取源，源和目标相同时直接跳过（否则会把自己删掉）。
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for position, record in enumerate(data["records"], start=1):
        record["index"] = position
        original = record.get("original_screenshot")
        if not original:
            original = str(record.get("screenshot", ""))
            record["original_screenshot"] = original
        source = Path(str(original))
        if not source.is_file():
            continue
        target = target_dir / f"frame_{position:04d}.png"
        if source.resolve() == target.resolve():
            record["screenshot"] = str(target).replace("\\", "/")
            count += 1
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        record["source_frame"] = source.name
        record["screenshot"] = str(target).replace("\\", "/")
        count += 1
    return count


def save_annotation(path: Path, data: dict[str, Any]) -> None:
    """写回 annotation JSON（保持缩进与中文原样）。"""

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pre_annotate(
    data: dict[str, Any],
    *,
    mode: str,
    weights: Path | None = None,
    conf: float = 0.25,
    overwrite: bool = False,
    temporal: bool = False,
) -> dict[str, int]:
    """给所有帧生成初始框并写入连续编号 `index`；返回统计。

    人工画过的框（`source == "human"`）永远保留：这类帧会把人工框与新生成的自动框
    合并，且合并结果标记为 human，避免重跑预标注冲掉人工成果。
    """

    records: list[dict[str, Any]] = data["records"]
    stats = {"frames": 0, "labels": 0, "skipped_existing": 0, "kept_human": 0}
    previous_frame: NDArray[np.uint8] | None = None
    for position, record in enumerate(records, start=1):
        record["index"] = position  # 连续编号：界面上的序号不再跳号
        if record.get("result") != "ok":
            continue
        frame_path = Path(record["screenshot"])
        if not frame_path.is_file():
            continue
        frame = read_bgr(frame_path)
        existing = record.get("labels") or []
        human = [item for item in existing if item.get("source") == "human"]
        if human:
            generated = _detect(mode, previous_frame, frame, weights, conf)
            record["labels"] = merge_labels([*human, *generated])
            record["auto_note"] = "human+auto（人工框优先保留）"
            stats["kept_human"] += 1
            previous_frame = frame
            continue
        if existing and not overwrite:
            stats["skipped_existing"] += 1
            previous_frame = frame
            continue
        labels = _detect(mode, previous_frame, frame, weights, conf)
        ignored = record.get("ignored_boxes") or []
        if ignored:
            labels = [
                item
                for item in labels
                if not any(_overlaps_ignored(item, region) for region in ignored)
            ]
        record["labels"] = labels
        if mode == "heuristic" and not labels:
            record["auto_note"] = "empty（全局运动帧或画面静止，交给人工确认）"
        else:
            record.pop("auto_note", None)
        stats["frames"] += 1
        stats["labels"] += len(labels)
        previous_frame = frame
    if temporal:
        stats["temporal"] = temporal_filter(data)
        stats["static"] = static_filter(data)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 2 标注工具（自动预标注 + 人工抽查）")
    parser.add_argument("--annotation", type=Path, required=True, help="annotation JSON 路径")
    parser.add_argument(
        "--pre-annotate", choices=("none", "heuristic", "yolo"), default="none", help="预标注方式"
    )
    parser.add_argument("--weights", type=Path, default=None, help="YOLO 权重（pre-annotate=yolo）")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--overwrite", action="store_true", help="预标注时覆盖已有 labels")
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="预标注后做时间一致性过滤：丢掉只在单帧出现的框（弹道/特效）",
    )
    parser.add_argument(
        "--unify-frames",
        action="store_true",
        help="整理成连续编号 frame_0001.png…（硬链接到 frames_unified，零拷贝）",
    )
    parser.add_argument("--no-gui", action="store_true", help="只做预标注，不开 GUI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_annotation(args.annotation)
    if args.pre_annotate == "none" and not args.unify_frames:
        return run_gui(args.annotation, data) if not args.no_gui else 0
    if args.pre_annotate != "none":
        stats = pre_annotate(
            data,
            mode=args.pre_annotate,
            weights=args.weights,
            conf=args.conf,
            overwrite=args.overwrite,
            temporal=args.temporal,
        )
        print(
            f"预标注完成（{args.pre_annotate}）：处理 {stats['frames']} 帧，"
            f"生成 {stats['labels']} 个框，跳过已有标注 {stats['skipped_existing']} 帧，"
            f"保留人工标注 {stats['kept_human']} 帧"
        )
    if args.unify_frames:
        target = args.annotation.parent / "frames_unified"
        count = renumber_frames(data, target)
        print(f"帧已统一编号：{count} 张 → {target}")
    save_annotation(args.annotation, data)
    if args.no_gui:
        return 0
    return run_gui(args.annotation, data)


def run_gui(annotation_path: Path, data: dict[str, Any]) -> int:
    """tkinter 抽查界面。"""

    import tkinter as tk
    from tkinter import messagebox

    from PIL import Image, ImageTk

    records: list[dict[str, Any]] = data["records"]
    viewable = [item for item in records if item.get("result") == "ok"]
    if not viewable:
        print("没有可查看的帧（result != ok）")
        return 1

    window = tk.Tk()
    window.title(f"标注抽查 - {annotation_path.name}")
    canvas = tk.Canvas(window, width=960, height=540, bg="#101010", highlightthickness=0)
    canvas.pack()
    status = tk.Label(window, anchor="w", font=("Consolas", 10))
    status.pack(fill="x")
    help_text = tk.Label(
        window,
        anchor="w",
        justify="left",
        font=("Consolas", 9),
        fg="#444",
        text=(
            "←/A 上一帧   →/D 下一帧   P/N 上/下一张待检查   "
            "X 删除选中框   R 重新框选   S 保存   空格 标记已检查   Q 退出"
        ),
    )
    help_text.pack(fill="x")

    state: dict[str, Any] = {
        "index": 0,
        "selected": None,
        "drawing": False,
        "dirty": False,
        "photo": None,
    }

    def current() -> dict[str, Any]:
        return viewable[state["index"]]

    def render() -> None:
        record = current()
        frame = read_bgr(Path(record["screenshot"]))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((960, 540))
        state["photo"] = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        scale_x, scale_y = 960 / frame.shape[1], 540 / frame.shape[0]
        for index, label in enumerate(record.get("labels") or []):
            x1, y1, x2, y2 = label["xyxy"]
            if label.get("ignored"):
                color = "#8a8a8a"
            elif index == state["selected"]:
                color = "#ff5252"
            else:
                color = "#00e5ff"
            width = 4 if index == state["selected"] else 3
            canvas.create_rectangle(
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
                outline=color,
                width=width,
                dash=(5, 3) if label.get("ignored") else None,
            )
            canvas.create_text(
                x1 * scale_x + 4,
                y1 * scale_y - 8,
                anchor="w",
                fill=color,
                font=("Consolas", 9),
                text=("忽略" if label.get("ignored") else f"{label['class_name']}#{index}"),
            )
        total = len(record.get("labels") or [])
        ignored_total = sum(1 for item in record.get("labels") or [] if item.get("ignored"))
        reviewed = "已检查" if record.get("reviewed") else "待检查"
        stats = record.get("stats") or {}
        state["dirty"] = state["dirty"]
        status.config(
            text=(
                f"[{state['index'] + 1}/{len(viewable)}] #{record.get('index')} "
                f"{Path(record['screenshot']).name}  "
                f"框 {total}（忽略 {ignored_total}）  {reviewed}  "
                f"亮度 {stats.get('mean_brightness')}  "
                f"模式 {'画框中(R=退出)' if state['drawing'] else '查看'}  "
                f"{'● 未保存' if state['dirty'] else ''}"
            )
        )

    def on_delete(*_: object) -> None:
        record = current()
        labels = record.get("labels") or []
        if state["selected"] is not None and 0 <= state["selected"] < len(labels):
            labels.pop(state["selected"])
            state["selected"] = None
            state["dirty"] = True
            render()

    def on_toggle_ignored(*_: object) -> None:
        """把选中的框标成"非敌人/忽略"，并把它记进 ignored_boxes 以便重跑预标注时不复活。"""

        record = current()
        labels = record.get("labels") or []
        if state["selected"] is None or not (0 <= state["selected"] < len(labels)):
            return
        box = labels[state["selected"]]
        box["ignored"] = not box.get("ignored", False)
        if box["ignored"]:
            record.setdefault("ignored_boxes", []).append(
                {"xyxy": list(box["xyxy"]), "reason": "human-ignore"}
            )
        else:
            record["ignored_boxes"] = [
                item
                for item in record.get("ignored_boxes", [])
                if item["xyxy"] != list(box["xyxy"])
            ]
        state["dirty"] = True
        render()

    def on_toggle_ignored_neighbours(*_: object) -> None:
        """把当前框和相邻帧同一位置的自动框一起标为忽略。

        弹道/技能特效通常连续出现在好多帧里，逐个按 E 太累；这里按 ±3 帧、中心 28px
        的邻域批量处理。人工框不会被自动标记。
        """

        record = current()
        labels = record.get("labels") or []
        if state["selected"] is None or not (0 <= state["selected"] < len(labels)):
            return
        target = labels[state["selected"]]
        on_toggle_ignored()

        marked = 0
        for offset in range(1, NEIGHBOUR_MARK_FRAMES + 1):
            for direction in (-1, 1):
                index = state["index"] + direction * offset
                if not (0 <= index < len(viewable)):
                    break
                neighbour = viewable[index]
                if neighbour.get("batch") != record.get("batch"):
                    break
                for box in neighbour.get("labels") or []:
                    if box.get("source") == "human" or box.get("ignored"):
                        continue
                    if _center_distance(box, target) <= TEMPORAL_RADIUS_PX:
                        box["ignored"] = True
                        neighbour.setdefault("ignored_boxes", []).append(
                            {"xyxy": list(box["xyxy"]), "reason": "human-ignore-neighbour"}
                        )
                        marked += 1
        state["dirty"] = True
        render()
        print(f"已把 {marked} 个相邻帧的同类框一起标为忽略")

    def linear(next_index: int) -> None:
        state["index"] = max(0, min(len(viewable) - 1, next_index))
        state["selected"] = None
        render()

    def find_unreviewed(step: int) -> None:
        index = state["index"] + step
        while 0 <= index < len(viewable):
            if not viewable[index].get("reviewed"):
                state["index"] = index
                state["selected"] = None
                render()
                return
            index += step

    def on_toggle_reviewed(*_: object) -> None:
        record = current()
        record["reviewed"] = not record.get("reviewed", False)
        state["dirty"] = True
        render()

    def on_save(*_: object) -> None:
        save_annotation(annotation_path, data)
        state["dirty"] = False
        render()
        print(f"已保存：{annotation_path}")

    def on_canvas_click(event: Any) -> None:
        record = current()
        labels = record.get("labels") or []
        x, y = event.x * 1280 / 960, event.y * 720 / 540
        state["selected"] = None
        for index, label in enumerate(labels):
            x1, y1, x2, y2 = label["xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["selected"] = index
                break
        render()

    def on_draw_start(event: Any) -> None:
        if not state["drawing"]:
            return
        state["drag_start"] = (event.x, event.y)

    def on_draw_end(event: Any) -> None:
        if not state["drawing"] or "drag_start" not in state:
            return
        x1, y1 = state["drag_start"]
        if abs(event.x - x1) < 5 or abs(event.y - y1) < 5:
            return
        record = current()
        box = [
            x1 * 1280 / 960,
            y1 * 720 / 540,
            event.x * 1280 / 960,
            event.y * 720 / 540,
        ]
        record.setdefault("labels", []).append(make_label(box, source="human"))
        state["selected"] = len(record["labels"]) - 1
        state["dirty"] = True
        render()

    def on_key(event: Any) -> None:
        action = key_action(event.keysym)
        if action == "prev_frame":
            linear(state["index"] - 1)
        elif action == "next_frame":
            linear(state["index"] + 1)
        elif action == "prev_unreviewed":
            find_unreviewed(-1)
        elif action == "next_unreviewed":
            find_unreviewed(1)
        elif action == "delete_box":
            on_delete()
        elif action == "toggle_ignored":
            on_toggle_ignored()
        elif action == "toggle_ignored_neighbours":
            on_toggle_ignored_neighbours()
        elif action == "toggle_draw":
            state["drawing"] = not state["drawing"]
            render()
        elif action == "save":
            on_save()
        elif action == "toggle_reviewed":
            on_toggle_reviewed()
        elif action == "quit":
            on_quit()

    def on_quit() -> None:
        if state["dirty"] and not messagebox.askyesno("未保存", "有改动未保存，确定退出？"):
            return
        window.destroy()

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<ButtonPress-1>", on_draw_start, add="+")
    canvas.bind("<ButtonRelease-1>", on_draw_end)
    window.bind("<Key>", on_key)
    window.protocol("WM_DELETE_WINDOW", on_quit)
    render()
    window.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

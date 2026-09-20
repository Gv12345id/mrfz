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

# 状态向量最多容纳 20 个敌人，预标注也按这个上限截断。
MAX_LABELS = 20

# 画面整体变化超过这个比例就视为"镜头平移/全局运动"帧：帧间差分会把背景边缘
# 全当成移动目标，与其产出噪声框，不如留空交给人工。
GLOBAL_MOTION_RATIO = 0.12

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


def merge_labels(
    labels: list[dict[str, Any]], *, iou_threshold: float = 0.3
) -> list[dict[str, Any]]:
    """合并高度重叠的候选框（保留面积较大的那个）。"""

    ordered = sorted(labels, key=_label_area, reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if all(label_iou(candidate, item) < iou_threshold for item in kept):
            kept.append(candidate)
    return kept[:MAX_LABELS]


def _label_area(label: dict[str, Any]) -> float:
    x1, y1, x2, y2 = label["xyxy"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _in_ui_mask(cx: float, cy: float) -> bool:
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in UI_MASKS)


def heuristic_labels(
    prev_frame: NDArray[np.uint8],
    frame: NDArray[np.uint8],
    *,
    pixel_threshold: int = 18,
) -> list[dict[str, Any]]:
    """用帧间差分找移动目标，作为敌人候选框（无权重时的预标注）。

    全局运动（镜头平移）帧直接返回空列表：那种帧上"动"的是整片背景，差分出来的
    框几乎全是噪声，让预标注留空比给错框更省人工。
    """

    if prev_frame.shape != frame.shape:
        return []
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(gray_prev, gray_cur)
    moved_ratio = float(np.count_nonzero(delta > pixel_threshold)) / delta.size
    if moved_ratio > GLOBAL_MOTION_RATIO:
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
        labels.append(make_label([x, y, x + w, y + h], source="auto"))
    return merge_labels(labels)


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
) -> dict[str, int]:
    """给所有帧生成初始框；返回统计。"""

    records: list[dict[str, Any]] = data["records"]
    stats = {"frames": 0, "labels": 0, "skipped_existing": 0}
    previous_frame: NDArray[np.uint8] | None = None
    for record in records:
        if record.get("result") != "ok":
            continue
        frame_path = Path(record["screenshot"])
        if not frame_path.is_file():
            continue
        frame = read_bgr(frame_path)
        existing = record.get("labels") or []
        if existing and not overwrite:
            stats["skipped_existing"] += 1
            previous_frame = frame
            continue
        if mode == "heuristic":
            labels = [] if previous_frame is None else heuristic_labels(previous_frame, frame)
        elif mode == "yolo":
            if weights is None:
                raise SystemExit("--pre-annotate yolo 需要同时给 --weights 指定权重文件")
            labels = yolo_labels(weights, frame, conf=conf)
        else:
            raise SystemExit(f"未知的预标注模式：{mode}")
        record["labels"] = labels
        if mode == "heuristic" and not labels:
            record["auto_note"] = "empty（全局运动帧或画面静止，交给人工确认）"
        else:
            record.pop("auto_note", None)
        stats["frames"] += 1
        stats["labels"] += len(labels)
        previous_frame = frame
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
    parser.add_argument("--no-gui", action="store_true", help="只做预标注，不开 GUI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_annotation(args.annotation)
    if args.pre_annotate != "none":
        stats = pre_annotate(
            data,
            mode=args.pre_annotate,
            weights=args.weights,
            conf=args.conf,
            overwrite=args.overwrite,
        )
        save_annotation(args.annotation, data)
        print(
            f"预标注完成（{args.pre_annotate}）：处理 {stats['frames']} 帧，"
            f"生成 {stats['labels']} 个框，跳过已有标注 {stats['skipped_existing']} 帧"
        )
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
            color = "#00e5ff" if index != state["selected"] else "#ff5252"
            width = 3 if index != state["selected"] else 4
            canvas.create_rectangle(
                x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, outline=color, width=width
            )
            canvas.create_text(
                x1 * scale_x + 4,
                y1 * scale_y - 8,
                anchor="w",
                fill=color,
                font=("Consolas", 9),
                text=f"{label['class_name']}#{index}",
            )
        total = len(record.get("labels") or [])
        reviewed = "已检查" if record.get("reviewed") else "待检查"
        stats = record.get("stats") or {}
        state["dirty"] = state["dirty"]
        status.config(
            text=(
                f"[{state['index'] + 1}/{len(viewable)}] {Path(record['screenshot']).name}  "
                f"框 {total}  {reviewed}  亮度 {stats.get('mean_brightness')}  "
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

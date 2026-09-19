"""图像读写与比较工具（BGR 约定，与 OpenCV 一致）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# BGR 图像，uint8，shape = (height, width, channels)
Frame = NDArray[np.uint8]


class ImageError(RuntimeError):
    """图像读取或写入失败。"""


def read_bgr(path: Path) -> Frame:
    """读取图片为 BGR 数组；失败时抛 :class:`ImageError`。"""

    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise ImageError(f"无法读取图片：{path}")
    return np.asarray(raw, dtype=np.uint8)


def write_bgr(path: Path, frame: Frame) -> None:
    """把 BGR 数组写为 PNG；失败时抛 :class:`ImageError`。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise ImageError(f"无法写入图片：{path}")


def size_of(frame: Frame) -> tuple[int, int]:
    """返回 ``(width, height)``。"""

    height, width = frame.shape[:2]
    return int(width), int(height)


def diff_ratio(before: Frame, after: Frame, *, pixel_threshold: int = 12) -> float:
    """返回发生明显变化的像素占比（0.0-1.0）。

    用于验证"点击后画面确实变了"。灰度差超过 ``pixel_threshold`` 记为变化像素。
    两张图尺寸不一致时直接返回 1.0（视为完全变化）。
    """

    if before.shape != after.shape:
        return 1.0
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(gray_before, gray_after)
    changed = int(np.count_nonzero(delta > pixel_threshold))
    return changed / float(delta.size)

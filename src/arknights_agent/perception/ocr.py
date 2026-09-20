"""费用 / 干员 CD 的数字识别。

实现方式：**模板匹配**，不依赖联网模型。

- 数字模板从战斗帧里自动提取（`tools/build_digit_templates.py`），属于游戏素材，
  落在 `.gitignore` 挡住的 `assets/templates/digit_templates.npz`，仓库里只有生成代码。
- 识别流程：裁 ROI → 按"白色且低饱和"阈值二值化 → 连通域按尺寸/长宽比过滤 →
  归一化到 24×32 → 与 0–9 模板比海明距离取最近 → 按 x 顺序拼成整数。
- PLANS.md 要求「双次读数一致才采信」，所以除了单帧 :func:`read_cost`，还提供
  :func:`read_cost_consensus`：连续两帧读数一致才返回，否则返回 None（宁可不动）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from arknights_agent.imaging import Frame

# 费用数字所在 ROI（x, y, w, h），1280×720 逻辑坐标系。
COST_ROI: tuple[int, int, int, int] = (1200, 487, 80, 50)

# 底部干员卡数字条（部署费用；干员进入冷却后同一位置显示冷却秒数，同字体）。
# 卡位是固定间距的网格：第 0 个数字左边缘约在 x=85，之后每张卡约 119.5px。
CARD_ROW_ROI: tuple[int, int, int, int] = (60, 596, 1220, 46)
CARD_SLOT_START_X = 85
CARD_SLOT_PITCH = 119.5
CARD_SLOT_COUNT = 10
CARD_SLOT_ROI = (-8, 2, 56, 40)  # 相对卡位左上角 (x, y, w, h)

# 同属一个数字的字形，水平间距不超过它（像素）。
GLYPH_GROUP_GAP = 8
# 分组判据：相邻字形左边缘间距超过它就认为是下一个数字（同一数字内约 22–28px，
# 不同卡片之间 80px 以上）。
GLYPH_GROUP_MAX_DELTA = 45

# 归一化后的数字尺寸 (宽, 高)，与模板库一致。
GLYPH_SIZE: tuple[int, int] = (24, 32)

MIN_GLYPH_HEIGHT = 20
MAX_GLYPH_HEIGHT = 46
MIN_GLYPH_WIDTH = 6
MAX_GLYPH_WIDTH = 34
# 底部卡片数字与费用数字同为一套字体、同一尺寸（实测 28×34 px）。
CARD_MIN_GLYPH_HEIGHT = 20
CARD_MAX_GLYPH_HEIGHT = 46
CARD_MIN_GLYPH_WIDTH = 6
CARD_MAX_GLYPH_WIDTH = 34
MIN_ASPECT = 0.25
MAX_ASPECT = 1.0

# 模板匹配的最大海明距离（24×32=768 位）：超过它说明这个字形不像任何数字，整帧读数作废。
# 实测同字体字形之间一般在 90 位以内，取 120 既容得下渲染差异，又能挡住"小碎块"误匹配。
MAX_MATCH_DISTANCE = 120

DEFAULT_TEMPLATE_PATH = Path("assets/templates/digit_templates.npz")

Mask = NDArray[np.bool_]


class OcrError(RuntimeError):
    """OCR 相关错误（模板缺失、读数不可信等）。"""


def extract_glyphs(image: Frame, roi: tuple[int, int, int, int] = COST_ROI) -> list[Mask]:
    """抠出 ROI 内的数字字形（按 x 从左到右），归一化到 :data:`GLYPH_SIZE`。"""

    return [glyph for _, glyph in _extract_glyph_boxes(image, roi)]


def _extract_glyph_boxes(
    image: Frame,
    roi: tuple[int, int, int, int],
    *,
    min_height: int = MIN_GLYPH_HEIGHT,
    max_height: int = MAX_GLYPH_HEIGHT,
    min_width: int = MIN_GLYPH_WIDTH,
    max_width: int = MAX_GLYPH_WIDTH,
) -> list[tuple[int, Mask]]:
    """同 :func:`extract_glyphs`，但尺寸范围可调，并额外返回字形的 x 坐标。"""

    x, y, w, h = roi
    patch = image[max(0, y) : y + h, max(0, x) : x + w]
    if patch.size == 0:
        return []
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 数字是白色（高亮度、低饱和）；面板底色暗，周围地图虽然亮但饱和度高。
    mask = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 80)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[int, Mask]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if not (min_height <= bh <= max_height):
            continue
        if not (min_width <= bw <= max_width):
            continue
        if not (MIN_ASPECT <= bw / bh <= MAX_ASPECT):
            continue
        glyph = mask[by : by + bh, bx : bx + bw]
        found.append((bx, cv2.resize(glyph, GLYPH_SIZE, interpolation=cv2.INTER_AREA) > 127))
    found.sort(key=lambda item: item[0])
    return found


@dataclass(frozen=True)
class DigitTemplates:
    """0–9 的数字模板（布尔图，shape = (10, 高, 宽)）。

    费用区与底部干员卡区**各有一套模板**：两处字号与抗锯齿不同，同一个数字在两处的
    海明距离实测可达 292 位（远超 120 的匹配阈值），共用一套均值模板会让两边都认不出来。
    `card_images` 缺失时卡片区退回费用模板——老的 npz（只有 `templates`）因此仍然可用。
    """

    images: NDArray[np.bool_]
    card_images: NDArray[np.bool_] | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_TEMPLATE_PATH) -> DigitTemplates:
        """从 npz 读取模板；缺失时给出可执行的修复建议。"""

        if not path.is_file():
            raise OcrError(
                f"数字模板不存在：{path}；先跑 "
                "`python tools/build_digit_templates.py`"
                "（带 --annotation 与 --labels）生成"
            )
        with np.load(path) as data:
            images = np.asarray(data["templates"], dtype=bool)
            cards = (
                np.asarray(data["card_templates"], dtype=bool) if "card_templates" in data else None
            )
        _check_template_shape(images, "templates", path)
        if cards is not None:
            _check_template_shape(cards, "card_templates", path)
        return cls(images, cards)

    @property
    def has_card_templates(self) -> bool:
        """是否带卡片区专用模板（假 = 卡片区退回费用模板）。"""

        return self.card_images is not None

    def _classify_with(self, bank: NDArray[np.bool_], glyph: Mask) -> tuple[int, int]:
        distances = [int(np.count_nonzero(glyph != template)) for template in bank]
        best = int(np.argmin(distances))
        return best, distances[best]

    def classify(self, glyph: Mask) -> tuple[int, int]:
        """费用区：返回 (数字, 海明距离)。"""

        return self._classify_with(self.images, glyph)

    def classify_card(self, glyph: Mask) -> tuple[int, int]:
        """卡片区：返回 (数字, 海明距离)；没有卡片模板时退回费用模板。"""

        bank = self.images if self.card_images is None else self.card_images
        return self._classify_with(bank, glyph)

    def read_number(
        self, glyphs: list[Mask], *, card: bool = False
    ) -> tuple[int | None, list[int]]:
        """把一串字形拼成整数；任一字形匹配太差就判为不可信（None）。

        `card=True` 用卡片区模板（见 :attr:`card_images`）。
        """

        if not glyphs:
            return None, []
        digits: list[int] = []
        for glyph in glyphs:
            digit, distance = self.classify_card(glyph) if card else self.classify(glyph)
            if distance > MAX_MATCH_DISTANCE:
                return None, digits
            digits.append(digit)
        return int("".join(str(digit) for digit in digits)), digits


def _check_template_shape(images: NDArray[np.bool_], name: str, path: Path) -> None:
    """模板必须是 (10, H, W)：少一个数字就等于静默认错，宁可报错。"""

    if images.ndim != 3 or images.shape[0] != 10:
        raise OcrError(f"{path} 的 {name} 形状应为 (10, H, W)，实际 {images.shape}")


def read_cost(
    frame: Frame, templates: DigitTemplates, roi: tuple[int, int, int, int] = COST_ROI
) -> int | None:
    """读一帧的费用值；无法可信识别时返回 None。"""

    value, _digits = templates.read_number(extract_glyphs(frame, roi))
    return value


def extract_card_glyphs(frame: Frame) -> list[Mask]:
    """按卡位网格抠出底部卡片上的数字字形（供模板提取与调试用）。"""

    glyphs: list[Mask] = []
    for slot in range(CARD_SLOT_COUNT):
        slot_x = CARD_SLOT_START_X + round(slot * CARD_SLOT_PITCH)
        dx, dy, width, height = CARD_SLOT_ROI
        slot_roi = (slot_x + dx, CARD_ROW_ROI[1] + dy, width, height)
        glyphs.extend(
            glyph
            for _, glyph in _extract_glyph_boxes(
                frame,
                slot_roi,
                min_height=CARD_MIN_GLYPH_HEIGHT,
                max_height=CARD_MAX_GLYPH_HEIGHT,
                min_width=CARD_MIN_GLYPH_WIDTH,
                max_width=CARD_MAX_GLYPH_WIDTH,
            )
        )
    return glyphs


def read_card_numbers(
    frame: Frame, templates: DigitTemplates, roi: tuple[int, int, int, int] = CARD_ROW_ROI
) -> list[int | None]:
    """读底部干员卡上的数字（待部署时是部署费用，冷却中是冷却秒数——同字体）。

    字形按"左边缘间距"分组：同一数字内相邻字形间距约 22–28px，不同卡片之间约 80px 以上，
    所以用中间值 :data:`GLYPH_GROUP_MAX_DELTA` 分段。返回从左到右的读数列表，
    某个数字拼不出来时该位为 None。

    这里用**卡片区模板**（`templates.card_images`）匹配；老 npz 没有卡片模板时退回费用模板，
    但费用模板对卡片字形的海明距离常常超过阈值，读数会大面积变成 None——那说明该重建模板了
    （`tools/build_digit_templates.py --source both`）。
    """

    results: list[int | None] = []
    for slot in range(CARD_SLOT_COUNT):
        slot_x = CARD_SLOT_START_X + round(slot * CARD_SLOT_PITCH)
        dx, dy, width, height = CARD_SLOT_ROI
        slot_roi = (slot_x + dx, roi[1] + dy, width, height)
        glyphs = [
            glyph
            for _, glyph in _extract_glyph_boxes(
                frame,
                slot_roi,
                min_height=CARD_MIN_GLYPH_HEIGHT,
                max_height=CARD_MAX_GLYPH_HEIGHT,
                min_width=CARD_MIN_GLYPH_WIDTH,
                max_width=CARD_MAX_GLYPH_WIDTH,
            )
        ]
        if not glyphs:
            results.append(None)
            continue
        results.append(templates.read_number(glyphs, card=True)[0])
    return results


def read_cost_consensus(
    frames: list[Frame],
    templates: DigitTemplates,
    roi: tuple[int, int, int, int] = COST_ROI,
) -> int | None:
    """连续两帧读数一致才采信（PLANS.md：双次读数一致才采信）。

    frames 至少两帧；取最后一对相邻帧比较。不一致或任一帧读不出就返回 None——
    状态不明确时上层应当等下一帧，而不是拿一个可能错的费用去做决策。
    """

    if len(frames) < 2:
        return None
    first = read_cost(frames[-2], templates, roi)
    second = read_cost(frames[-1], templates, roi)
    if first is None or second is None or first != second:
        return None
    return second

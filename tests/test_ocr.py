"""`perception/ocr.py` 的测试：字形提取、模板匹配、双次读数一致。

用 OpenCV 自带字体渲染合成数字来构造模板与画面，因此不依赖任何游戏素材。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from arknights_agent.imaging import Frame
from arknights_agent.perception.ocr import (
    CARD_ROW_ROI,
    COST_ROI,
    DigitTemplates,
    OcrError,
    extract_glyphs,
    read_card_numbers,
    read_cost,
    read_cost_consensus,
)

GLYPH_W, GLYPH_H = 24, 32
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _base_digit(digit: int) -> np.ndarray:
    """数字的基础位图（模板与画面都由它缩放而来，保证形状一致）。"""

    canvas = np.zeros((28, 20), dtype=np.uint8)
    cv2.putText(canvas, str(digit), (1, 23), FONT, 0.8, 255, 2)
    return (canvas > 127).astype(bool)


def _synthetic_templates() -> DigitTemplates:
    """模板直接用"每帧一个数字"经同一套提取管道得到，避免测到字体缩放差异。"""

    glyphs = []
    for digit in range(10):
        extracted = extract_glyphs(_battle_frame(digit))
        assert len(extracted) == 1, f"数字 {digit} 应当只抠出一个字形"
        glyphs.append(extracted[0])
    return DigitTemplates(np.stack(glyphs))


def _battle_frame(cost: int) -> Frame:
    """画一张带费用数字的假战斗帧（数字位置与真实 ROI 对齐）。"""

    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    x, y, _w, _h = COST_ROI
    for index, char in enumerate(str(cost)):
        glyph = (_base_digit(int(char)) * 255).astype(np.uint8)
        glyph = cv2.resize(glyph, (30, 42), interpolation=cv2.INTER_NEAREST)
        top = y + 4
        left = x + 2 + index * 36
        frame[top : top + 42, left : left + 30] = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
    return frame


def test_extract_glyphs_finds_white_digits_only() -> None:
    frame = _battle_frame(76)
    glyphs = extract_glyphs(frame)

    assert len(glyphs) == 2
    assert all(glyph.shape == (GLYPH_H, GLYPH_W) for glyph in glyphs)


def test_read_cost_reads_single_and_double_digit() -> None:
    templates = _synthetic_templates()

    assert read_cost(_battle_frame(7), templates) == 7
    assert read_cost(_battle_frame(76), templates) == 76
    assert read_cost(_battle_frame(0), templates) == 0


def test_read_cost_consensus_requires_two_matching_frames() -> None:
    templates = _synthetic_templates()

    assert read_cost_consensus([_battle_frame(42), _battle_frame(42)], templates) == 42
    assert read_cost_consensus([_battle_frame(42), _battle_frame(43)], templates) is None
    assert read_cost_consensus([_battle_frame(42)], templates) is None


def test_read_cost_returns_none_when_nothing_recognisable() -> None:
    templates = _synthetic_templates()
    blank = np.full((720, 1280, 3), 20, dtype=np.uint8)

    assert read_cost(blank, templates) is None


def test_missing_template_file_raises_with_hint(tmp_path: Path) -> None:
    try:
        DigitTemplates.load(tmp_path / "nope.npz")
    except OcrError as exc:
        assert "build_digit_templates" in str(exc)
    else:  # pragma: no cover - 只要抛错就走到这里
        raise AssertionError("模板缺失时应当抛 OcrError")


def test_read_number_rejects_far_glyph() -> None:
    templates = _synthetic_templates()
    # 一条横杠：形状离任何数字都很远
    noise = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
    noise[14:18, 2:22] = True

    value, digits = templates.read_number([noise])

    assert value is None
    assert digits == []


def test_load_reads_both_template_banks(tmp_path: Path) -> None:
    """npz 里同时有 templates 与 card_templates 时两套都要读进来。"""

    cost = np.stack([_base_digit(digit) for digit in range(10)])
    cards = np.roll(cost, 1, axis=0)
    path = tmp_path / "templates.npz"
    np.savez_compressed(path, templates=cost, card_templates=cards)

    templates = DigitTemplates.load(path)

    assert templates.has_card_templates is True
    assert templates.card_images is not None
    assert np.array_equal(templates.card_images, cards)


def test_load_without_card_bank_falls_back_to_cost(tmp_path: Path) -> None:
    """老 npz 只有 templates：卡片区退回费用模板，不报错。"""

    cost = np.stack([_base_digit(digit) for digit in range(10)])
    path = tmp_path / "templates.npz"
    np.savez_compressed(path, templates=cost)

    templates = DigitTemplates.load(path)

    assert templates.has_card_templates is False
    assert templates.classify_card(cost[4]) == templates.classify(cost[4])


def test_load_rejects_wrong_card_bank_shape(tmp_path: Path) -> None:
    path = tmp_path / "templates.npz"
    np.savez_compressed(
        path,
        templates=np.stack([_base_digit(digit) for digit in range(10)]),
        card_templates=np.zeros((9, GLYPH_H, GLYPH_W), dtype=bool),
    )

    try:
        DigitTemplates.load(path)
    except OcrError as exc:
        assert "card_templates" in str(exc)
    else:  # pragma: no cover - 只要抛错就走到这里
        raise AssertionError("卡片模板形状不对时应当抛 OcrError")


def test_card_reads_use_the_card_bank() -> None:
    """卡片读数必须走 card_images：把卡片模板整体错位一位，读数就该跟着变。

    这一条是 T3 的核心回归——费用模板与卡片模板不能互相顶替。
    """

    cost = np.stack([_base_digit(digit) for digit in range(10)])
    templates = DigitTemplates(cost, np.roll(cost, 1, axis=0))

    for digit in range(10):
        cost_value, _ = templates.read_number([cost[digit]])
        card_value, _ = templates.read_number([cost[digit]], card=True)
        assert cost_value == digit
        assert card_value == (digit + 1) % 10


def _card_frame(digit: int, *, slot: int = 0) -> Frame:
    """在指定卡位画一个数字的假战斗帧（位置照 ocr.CARD_SLOT_ROI 的几何）。"""

    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    glyph = (_base_digit(digit) * 255).astype(np.uint8)
    glyph = cv2.resize(glyph, (30, 42), interpolation=cv2.INTER_NEAREST)
    left = 85 + round(slot * 119.5) + 5
    top = CARD_ROW_ROI[1] + 4
    frame[top : top + 42, left : left + 30] = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
    return frame


def test_read_card_numbers_reads_slot_zero_via_card_bank() -> None:
    # 模板必须与现场字形走同一条提取管道（同样是 24×32 归一化后的位图）。
    bank = np.stack([extract_glyphs(_battle_frame(digit))[0] for digit in range(10)])
    templates = DigitTemplates(bank, bank.copy())

    numbers = read_card_numbers(_card_frame(7), templates)

    assert len(numbers) == 10
    assert numbers[0] == 7

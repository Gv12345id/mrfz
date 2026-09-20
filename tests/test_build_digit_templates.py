"""`tools/build_digit_templates.py` 的纯逻辑测试（不读游戏素材、不跑推理）。

重点是 T3 的两区域模板：费用区与卡片区各聚各类、各写各的 npz key，
且只重建卡片区时不能把已有的费用模板冲掉。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BUILD_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_digit_templates.py"
GLYPH_H, GLYPH_W = 32, 24


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_digit_templates", BUILD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _bank(offset: int = 0) -> np.ndarray:
    """10 张可区分的假模板（不用真字体，只验证搬运逻辑）。"""

    bank = np.zeros((10, GLYPH_H, GLYPH_W), dtype=bool)
    for digit in range(10):
        bank[digit, digit + 2 : digit + 6, 4:20] = True
    return np.roll(bank, offset, axis=0)


def test_parse_labels_handles_none_and_commas() -> None:
    assert builder.parse_labels(None) is None
    assert builder.parse_labels("7,6,1") == [7, 6, 1]
    assert builder.parse_labels("7, 6 , 1,") == [7, 6, 1]


def test_write_templates_cost_only(tmp_path: Path) -> None:
    path = tmp_path / "templates.npz"

    builder.write_templates(
        path, {"cost": _bank()}, {"cost": {"labels": list(range(10)), "sizes": [1] * 10}}, frames=96
    )

    with np.load(path) as data:
        assert "templates" in data
        assert "card_templates" not in data
        assert int(data["source_frames"]) == 96
        assert list(data["cluster_labels"]) == list(range(10))


def test_write_templates_cards_only_keeps_existing_cost_bank(tmp_path: Path) -> None:
    """只重建卡片区时必须保留原费用模板——npz 是整文件覆盖，不合并就会冲掉。"""

    path = tmp_path / "templates.npz"
    cost = _bank()
    np.savez_compressed(path, templates=cost)

    builder.write_templates(
        path,
        {"cards": _bank(1)},
        {"cards": {"labels": list(range(10)), "sizes": [2] * 10}},
        frames=50,
    )

    with np.load(path) as data:
        assert np.array_equal(data["templates"], cost)
        assert np.array_equal(data["card_templates"], _bank(1))
        assert list(data["cards_cluster_labels"]) == list(range(10))


def test_write_templates_cards_only_without_cost_bank_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="没有费用模板"):
        builder.write_templates(
            tmp_path / "templates.npz",
            {"cards": _bank()},
            {"cards": {"labels": list(range(10)), "sizes": [1] * 10}},
            frames=10,
        )


def test_write_templates_both_regions(tmp_path: Path) -> None:
    path = tmp_path / "templates.npz"

    builder.write_templates(
        path,
        {"cost": _bank(), "cards": _bank(1)},
        {
            "cost": {"labels": list(range(10)), "sizes": [3] * 10},
            "cards": {"labels": list(range(10)), "sizes": [4] * 10},
        },
        frames=120,
    )

    with np.load(path) as data:
        assert {"templates", "card_templates"} <= set(data.files)
        # 费用沿用旧键名（向后兼容），卡片用 cards_ 前缀。
        assert list(data["cluster_sizes"]) == [3] * 10
        assert list(data["cards_cluster_sizes"]) == [4] * 10

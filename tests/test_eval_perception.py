"""`tools/eval_perception.py` 的纯逻辑测试（不加载权重、不跑推理）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

EVAL_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_perception.py"


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location("eval_perception", EVAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_eval()


def test_p95_uses_nearest_rank_like_verify() -> None:
    """与 scripts/verify.py 同口径：n=100 取第 95 位。"""

    assert evaluator.p95_ms([float(value) for value in range(1, 101)]) == 95.0
    assert evaluator.p95_ms([5.0]) == 5.0
    with pytest.raises(ValueError, match="样本为空"):
        evaluator.p95_ms([])


def test_presence_scores_counts_missed_frames_as_false_negative() -> None:
    truth = [True, True, True, False, False]
    predicted = [True, False, False, True, False]

    scores = evaluator.presence_scores(truth, predicted)

    assert scores["frames_with_enemy"] == 3
    assert scores["true_positive"] == 1
    assert scores["false_negative"] == 2
    assert scores["false_positive"] == 1
    assert scores["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert scores["precision"] == pytest.approx(0.5)


def test_presence_scores_all_positive() -> None:
    scores = evaluator.presence_scores([True, True], [True, True])

    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0


def test_presence_scores_handles_no_positive_truth() -> None:
    """真值里一帧敌人都没有时，召回按 0 记（不除以 0），由门禁去判红。"""

    scores = evaluator.presence_scores([False, False], [False, True])

    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0
    assert scores["false_positive"] == 1


def test_presence_scores_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="数量不一致"):
        evaluator.presence_scores([True], [True, False])
    with pytest.raises(ValueError, match="样本为空"):
        evaluator.presence_scores([], [])


def test_label_has_box_reads_yolo_label_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    filled = tmp_path / "filled.txt"
    filled.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    assert evaluator.label_has_box(empty) is False
    assert evaluator.label_has_box(filled) is True
    assert evaluator.label_has_box(tmp_path / "missing.txt") is False

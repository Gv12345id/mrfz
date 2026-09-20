"""holdout 集上的指标门禁：mAP / 召回 / 状态维度 / 延迟 / OCR 准确率。

对应 PLANS.md 阶段 2 验收 2，阈值与 `scripts/verify.py --phase 2` 的 check 一一对应
（同一份产物、同一条阈值，改阈值要同时改 PLANS.md）。

**产物缺失时整模块 skip，并写明原因**：本测试要的是 T2 重训后由评测步骤产出的
`runs/phase2/perception_metrics.json` 与 `runs/phase2/ocr_report.json`，离线单测环境里没有。
这不算放行——`verify.py --phase 2` 的 `enemy-map50` / `ocr-cost-accuracy` 等 check
读同一份产物，缺产物时**如实判 fail**。两者合起来才是完整的门禁。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arknights_agent.perception.state import STATE_DIM

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_METRICS = ROOT / "runs" / "phase2" / "perception_metrics.json"
OCR_REPORT = ROOT / "runs" / "phase2" / "ocr_report.json"

# 与 scripts/verify.py 的常量保持一致（PLANS.md 阶段 2 验收标准）。
ENEMY_MAP50_THRESHOLD = 0.75
ENEMY_RECALL_THRESHOLD = 0.90
OCR_ACCURACY_THRESHOLD = 0.98
PERCEPTION_LATENCY_P95_THRESHOLD_MS = 300.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def perception_metrics() -> dict[str, Any]:
    if not PERCEPTION_METRICS.is_file():
        pytest.skip(
            f"缺少 {PERCEPTION_METRICS.relative_to(ROOT)}（T2 重训后的评测产物）；"
            "verify.py --phase 2 的 enemy-map50 检查会因此判 fail"
        )
    return _load(PERCEPTION_METRICS)


@pytest.fixture(scope="module")
def ocr_report() -> dict[str, Any]:
    if not OCR_REPORT.is_file():
        pytest.skip(f"缺少 {OCR_REPORT.relative_to(ROOT)}")
    return _load(OCR_REPORT)


def test_enemy_map50_meets_threshold(perception_metrics: dict[str, Any]) -> None:
    assert perception_metrics["enemy_map50"] >= ENEMY_MAP50_THRESHOLD


def test_enemy_presence_recall_meets_threshold(perception_metrics: dict[str, Any]) -> None:
    """「战场存在敌人」的召回：有敌人的帧不能被整帧判空。"""

    assert perception_metrics["enemy_recall"] >= ENEMY_RECALL_THRESHOLD


def test_state_dim_matches_contract(perception_metrics: dict[str, Any]) -> None:
    assert perception_metrics["state_dim"] == STATE_DIM


def test_perception_latency_p95_within_budget(perception_metrics: dict[str, Any]) -> None:
    assert perception_metrics["latency_p95_ms"] <= PERCEPTION_LATENCY_P95_THRESHOLD_MS


def test_cost_ocr_accuracy_meets_threshold(ocr_report: dict[str, Any]) -> None:
    assert ocr_report["accuracy"] >= OCR_ACCURACY_THRESHOLD


def test_cd_ocr_accuracy_meets_threshold(ocr_report: dict[str, Any]) -> None:
    """干员 CD 数字：T3 采到冷却样本后才有 `cd_field.accuracy`。"""

    accuracy = (ocr_report.get("cd_field") or {}).get("accuracy")
    if accuracy is None:
        pytest.skip(
            "ocr_report.json 里还没有 cd_field.accuracy（T3 冷却样本未采集）；"
            "verify.py --phase 2 的 ocr-cd-accuracy 检查会因此判 fail"
        )
    assert accuracy >= OCR_ACCURACY_THRESHOLD

"""`scripts/verify.py` 的 JSON 契约与聚合逻辑测试。

verify.py 是阶段门禁的唯一放行判据，CI 与阶段 4 的循环都只读它的 JSON，
所以契约本身要有测试兜住：字段名、`passed` 的语义、退出码、覆盖率聚合、
schema hash 的稳定性。
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

VERIFY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify.py"


def _load_verify() -> Any:
    spec = importlib.util.spec_from_file_location("verify_script", VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses 需要模块已注册在 sys.modules 里才能解析 cls.__module__。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()

EXPECTED_PHASE1_CHECKS = (
    "static-checks",
    "offline-tests",
    "coverage-mcp",
    "coverage-device",
    "schema-doc",
    "schema-hash",
    "doctor-json",
    "device-loop-tests",
    "screenshot-failures",
    "screenshot-p95-ms",
    "click-loop-successes",
)

EXPECTED_PHASE2_CHECKS = (
    "static-checks",
    "offline-tests",
    "perception-metrics-tests",
    "state-vector-tests",
    "enemy-map50",
    "enemy-recall",
    "ocr-cost-accuracy",
    "ocr-cd-accuracy",
    "state-dim",
    "perception-latency-p95-ms",
    "perception-fixtures-hash",
    "device-state-legal-rate",
)


def test_phase1_check_names_match_plans_acceptance_criteria() -> None:
    """阶段 1 的 check 名一旦改动，PLANS.md 的映射表必须同步。"""

    assert tuple(name for name, _ in verify.PHASE_CHECKS[1]) == EXPECTED_PHASE1_CHECKS


def test_phase2_check_names_match_plans_acceptance_criteria() -> None:
    """阶段 2 的 check 名一旦改动，PLANS.md 的映射表必须同步。"""

    assert tuple(name for name, _ in verify.PHASE_CHECKS[2]) == EXPECTED_PHASE2_CHECKS


def test_phase2_thresholds_match_plans() -> None:
    """阈值是门禁的一部分，改它必须同时改 PLANS.md。"""

    assert verify.ENEMY_MAP50_THRESHOLD == 0.75
    assert verify.ENEMY_RECALL_THRESHOLD == 0.90
    assert verify.OCR_ACCURACY_THRESHOLD == 0.98
    assert verify.STATE_VECTOR_DIM == 173
    assert verify.PERCEPTION_LATENCY_P95_THRESHOLD_MS == 300.0
    assert verify.DEVICE_STATE_FRAMES == 50


def test_phase2_missing_reports_fail_instead_of_passing(tmp_path: Path) -> None:
    """产物缺失时必须判 fail 并写清原因，不能因为读不到就返回通过。"""

    missing = tmp_path / "perception_metrics.json"
    value, problem = verify.read_json_number(missing, "enemy_map50")

    assert value is None
    assert "缺少" in problem


def test_report_matches_plans_json_contract() -> None:
    started_at = datetime(2026, 9, 19, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    checks = [
        verify.CheckResult("static-checks", True, "ok"),
        verify.CheckResult("screenshot-p95-ms", True, "", 412.0, 1500),
    ]

    report = verify.build_report(1, checks, started_at)

    assert set(report) == {"phase", "passed", "started_at", "checks"}
    assert report["phase"] == 1
    assert report["passed"] is True
    assert report["started_at"] == "2026-09-19T10:00:00+08:00"
    assert set(report["checks"][0]) == {"name", "passed", "detail", "value", "threshold"}
    assert report["checks"][0]["value"] is None
    assert report["checks"][1]["value"] == 412.0
    assert report["checks"][1]["threshold"] == 1500
    assert json.loads(json.dumps(report)) == report


def test_passed_is_logical_and_of_checks() -> None:
    started_at = datetime.now(UTC)
    ok = verify.CheckResult("a", True)
    failed = verify.CheckResult("b", False, "boom")

    assert verify.build_report(1, [ok, ok], started_at)["passed"] is True
    assert verify.build_report(1, [ok, failed], started_at)["passed"] is False


def test_exit_code_is_nonzero_when_any_check_fails() -> None:
    started_at = datetime.now(UTC)
    passing = verify.build_report(1, [verify.CheckResult("a", True)], started_at)
    failing = verify.build_report(1, [verify.CheckResult("a", False)], started_at)

    assert verify.exit_code(passing) == 0
    assert verify.exit_code(failing) != 0


def test_unimplemented_phase_and_unknown_check_exit_with_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """阶段 2-4 未实现、check 名写错时不能假装成功。"""

    assert verify.main(["--phase", "4"]) == 2
    capsys.readouterr()

    assert verify.main(["--phase", "1", "--only", "not-a-check"]) == 2
    assert "not-a-check" in capsys.readouterr().err


def test_aggregate_coverage_sums_covered_over_statements() -> None:
    coverage = {
        "files": {
            "src\\arknights_agent\\mcp\\client.py": {
                "summary": {"covered_lines": 80, "num_statements": 100}
            },
            "src/arknights_agent/mcp/tools.py": {
                "summary": {"covered_lines": 20, "num_statements": 100}
            },
            "src/arknights_agent/device/fake.py": {
                "summary": {"covered_lines": 9, "num_statements": 10}
            },
        }
    }

    assert verify.aggregate_coverage(coverage, "mcp") == 50.0
    assert verify.aggregate_coverage(coverage, "device") == 90.0


def test_aggregate_coverage_without_matching_files_is_none() -> None:
    assert verify.aggregate_coverage({"files": {}}, "mcp") is None
    assert verify.aggregate_coverage({}, "device") is None


def test_canonical_schema_hash_ignores_order_and_tracks_schema_changes() -> None:
    tools = [
        {
            "name": "click",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
        {"name": "screencap", "inputSchema": {"type": "object", "properties": {}}},
    ]
    reordered = list(reversed(tools))
    changed = [
        tools[0],
        {"name": "screencap", "inputSchema": {"type": "object", "properties": {"zoom": {}}}},
    ]

    digest = verify.canonical_schema_hash(tools)

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert verify.canonical_schema_hash(reordered) == digest
    assert verify.canonical_schema_hash(changed) != digest


def test_documented_schema_hash_reads_hash_from_doc() -> None:
    text = "schema_sha256: 144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a"

    assert (
        verify.documented_schema_hash(text)
        == "144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a"
    )
    assert verify.documented_schema_hash("没有 hash 的文档") is None


def test_p95_uses_nearest_rank() -> None:
    assert verify.p95_ms([float(value) for value in range(1, 101)]) == 95.0
    assert verify.p95_ms([7.5]) == 7.5
    assert verify.p95_ms([3.0, 1.0, 2.0]) == 3.0
    with pytest.raises(ValueError, match="样本为空"):
        verify.p95_ms([])


def test_number_field_rejects_bool_and_missing() -> None:
    assert verify.number_field({"p95_ms": 12.5}, "p95_ms") == 12.5
    assert verify.number_field({"ok": True}, "ok") is None
    assert verify.number_field({}, "p95_ms") is None


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_parse_sha256_manifest_skips_blanks_and_comments() -> None:
    text = f"# holdout 帧清单\n\n{DIGEST_A}  frame_0001.png\n{DIGEST_B}\tframe_0002.png\n"

    assert verify.parse_sha256_manifest(text) == [
        (DIGEST_A, "frame_0001.png"),
        (DIGEST_B, "frame_0002.png"),
    ]


def test_parse_sha256_manifest_lowercases_digest() -> None:
    assert verify.parse_sha256_manifest(f"{'A' * 64}  frame_0001.png") == [
        (DIGEST_A, "frame_0001.png")
    ]


def test_parse_sha256_manifest_rejects_broken_lines() -> None:
    """清单写坏了要报错，不能静默少校验几个文件。"""

    with pytest.raises(ValueError, match="第 1 行"):
        verify.parse_sha256_manifest("不是摘要  frame_0001.png")
    with pytest.raises(ValueError, match="第 1 行"):
        verify.parse_sha256_manifest(f"{DIGEST_A}  ")
    with pytest.raises(ValueError, match="没有任何条目"):
        verify.parse_sha256_manifest("# 只有注释\n")


def test_nested_number_walks_mapping_path() -> None:
    assert verify.nested_number({"cd_field": {"accuracy": 0.5}}, "cd_field", "accuracy") == 0.5
    assert verify.nested_number({"cd_field": {"accuracy": True}}, "cd_field", "accuracy") is None
    assert verify.nested_number({"cd_field": "pending"}, "cd_field", "accuracy") is None
    assert verify.nested_number({}, "cd_field", "accuracy") is None


def test_verify_manifest_checks_digest_and_reports_missing(tmp_path: Path) -> None:
    present = tmp_path / "frame_0001.png"
    present.write_bytes(b"frame-1")
    digest = verify.file_sha256(present)
    entries = [
        (digest, "frame_0001.png"),
        (DIGEST_B, "frame_0002.png"),
        (DIGEST_A, "frame_0001.png"),
    ]

    checked, problems = verify.verify_manifest(tmp_path, entries)

    assert checked == 1
    assert any("缺少 frame_0002.png" in problem for problem in problems)
    assert any("摘要不一致" in problem for problem in problems)


def test_verify_manifest_accepts_matching_digest(tmp_path: Path) -> None:
    target = tmp_path / "frame_0009.png"
    target.write_bytes(b"frame-9")

    checked, problems = verify.verify_manifest(
        tmp_path, [(verify.file_sha256(target), "frame_0009.png")]
    )

    assert (checked, problems) == (1, [])

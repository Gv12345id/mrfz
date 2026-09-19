"""`tools/collect_frames.py` 的记录格式与汇总逻辑测试（纯函数，不碰设备）。"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

COLLECT_PATH = Path(__file__).resolve().parents[1] / "tools" / "collect_frames.py"


def _load_collect() -> Any:
    spec = importlib.util.spec_from_file_location("collect_frames", COLLECT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_collect()
PLAYER_JSONL_FIELDS = {
    "step",
    "action",
    "mcp_tool",
    "params",
    "result",
    "elapsed_ms",
    "screenshot",
}


def _record(step: int, *, result: str = "ok", elapsed_ms: float = 50.0) -> dict[str, Any]:
    return collect.frame_record(
        step=step,
        screenshot=f"runs/phase2/frames/frame_{step:04d}.png",
        resolution="1280x720",
        sha256="a" * 64,
        captured_at=datetime(2026, 9, 19, 22, 0, tzinfo=timezone(timedelta(hours=8))),
        elapsed_ms=elapsed_ms,
        controller_id="ctrl_test",
        screen_context="battle",
        result=result,
    )


def test_record_keeps_player_jsonl_field_names() -> None:
    """逐帧记录必须保住阶段 1 player.jsonl 的同名字段，方便工具复用。"""

    record = _record(3)

    assert PLAYER_JSONL_FIELDS <= set(record)
    assert record["step"] == 3
    assert record["action"] == "SCREENCAP"
    assert record["mcp_tool"] == "screencap"
    assert record["params"]["resolution"] == 720
    assert record["result"] == "ok"
    assert record["labels"] == []
    assert record["screen_context"] == "battle"


def test_summarize_counts_failures_and_latency() -> None:
    records = [
        _record(1, elapsed_ms=40.0),
        _record(2, elapsed_ms=60.0),
        _record(3, result="fail:X"),
    ]

    meta = collect.summarize(
        records,
        device_name="fake",
        resolution="1280x720",
        screen_context="battle",
        interval_ms=500,
    )

    assert meta["frames"] == 3
    assert meta["failures"] == 1
    assert meta["read_only"] is True
    assert meta["elapsed_ms_mean"] == 50.0
    assert meta["elapsed_ms_p95"] == 60.0


def test_write_annotation_emits_json_jsonl_and_sha256(tmp_path: Path) -> None:
    records = [_record(1), _record(2)]
    meta = collect.summarize(
        records,
        device_name="fake",
        resolution="1280x720",
        screen_context="main",
        interval_ms=500,
    )

    json_path, jsonl_path, sha_path = collect.write_annotation(
        records, meta, out_dir=tmp_path, name="annotation_2"
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["frames"] == 2
    assert len(payload["records"]) == 2
    assert payload["records"][0]["step"] == 1
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    sha_lines = sha_path.read_text(encoding="utf-8").splitlines()
    assert sha_lines[0].endswith("frame_0001.png")
    assert sha_lines[0].startswith("a" * 64)

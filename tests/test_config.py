"""配置加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from arknights_agent.config import load_settings


def test_defaults_when_no_config_files(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    assert settings.mcp.command == "maa-mcp"
    assert settings.device.logical_width == 1280
    assert settings.device.logical_height == 720
    assert settings.device.match is None


def test_local_config_overrides_main(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "mcp:\n  command: maa-mcp\n  call_timeout_s: 30\ndevice:\n  match: 'MuMu'\n",
        encoding="utf-8",
    )
    (tmp_path / "config.local.yaml").write_text(
        "device:\n  match: '127.0.0.1:16416'\n", encoding="utf-8"
    )

    settings = load_settings(tmp_path)

    assert settings.mcp.call_timeout_s == 30
    assert settings.device.match == "127.0.0.1:16416"


def test_invalid_toplevel_raises(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层必须是映射"):
        load_settings(tmp_path)

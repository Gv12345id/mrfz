"""运行配置：MCP server 启动方式、目标设备与逻辑坐标系。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# 配置文件位置：config.yaml 入库，config.local.yaml 为本机私有覆盖（不入库）。
DEFAULT_CONFIG_DIR = Path("config")
CONFIG_FILENAME = "config.yaml"
LOCAL_CONFIG_FILENAME = "config.local.yaml"


class McpServerConfig(BaseModel):
    """MCP server（MaaMCP）的 stdio 启动方式。"""

    command: str = Field(default="maa-mcp", description="可执行文件或命令")
    args: tuple[str, ...] = Field(default=(), description="启动参数")
    env: dict[str, str] = Field(default_factory=dict, description="追加到子进程环境变量")
    startup_timeout_s: float = Field(
        default=180.0, gt=0, description="握手超时（首次握手会初始化 MaaFramework）"
    )
    call_timeout_s: float = Field(default=90.0, gt=0, description="单次工具调用超时")
    startup_retries: int = Field(
        default=2, ge=0, description="握手失败后的重试次数（退避 0.5s / 1s / ...）"
    )


class DeviceConfig(BaseModel):
    """目标设备与逻辑坐标系。

    ``match`` 是设备名/地址的匹配片段（MaaMCP 的 ``find_adb_device_list`` 返回的
    名称通常形如 ``MuMuEmulator12-127.0.0.1:16416``）。留空时只在恰好发现一台
    设备的情况下自动选中，发现多台则报错并要求显式配置。
    """

    match: str | None = None
    logical_width: int = Field(default=1280, gt=0)
    logical_height: int = Field(default=720, gt=0)
    click_duration_ms: int = Field(default=50, ge=0, description="点击按下时长")


class IntegrationConfig(BaseModel):
    """真机集成测试的安全闸门与点击目标。

    这几项只写在 ``config/config.local.yaml``（不入库）：

    - ``confirm_screen``：人工确认游戏已处于中立界面（不在关卡内）后才置 true，
      否则集成测试直接失败退出，绝不替人判断"现在可以点了"。
    - ``tap_a`` / ``tap_b``：一对"点开 / 收起"的已知元素逻辑坐标，
      集成测试靠它们交替点击并验证画面变化。
    """

    confirm_screen: bool = Field(default=False, description="人工确认游戏处于中立界面后才允许点击")
    tap_a: tuple[int, int] | None = Field(default=None, description="点击目标 A（逻辑坐标）")
    tap_b: tuple[int, int] | None = Field(default=None, description="点击目标 B（逻辑坐标）")


class Settings(BaseModel):
    """完整运行配置。"""

    mcp: McpServerConfig = Field(default_factory=McpServerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    integration: IntegrationConfig = Field(default_factory=lambda: IntegrationConfig())


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是映射：{path}")
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_settings(config_dir: Path | None = None) -> Settings:
    """读取配置目录下的 config.yaml 与 config.local.yaml（后者覆盖前者）。

    两个文件都不存在时返回全默认值。
    """

    directory = config_dir if config_dir is not None else DEFAULT_CONFIG_DIR
    data = _deep_merge(
        _read_yaml(directory / CONFIG_FILENAME),
        _read_yaml(directory / LOCAL_CONFIG_FILENAME),
    )
    return Settings.model_validate(data)

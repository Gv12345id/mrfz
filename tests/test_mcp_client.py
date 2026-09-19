"""MCP 客户端测试：结果解析 + 与离线 MCP server 的完整往返。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from arknights_agent.config import DeviceConfig, McpServerConfig
from arknights_agent.device.maa_via_mcp import MaaMcpDevice
from arknights_agent.imaging import diff_ratio, size_of
from arknights_agent.mcp.client import McpStartupError, McpStdioClient, McpToolError
from arknights_agent.mcp.tools import first_text, result_value

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


def echo_server_config(tmp_path: Path, **env: str) -> McpServerConfig:
    """构造一个指向离线噪声替身的客户端配置。"""

    return McpServerConfig(
        command=sys.executable,
        args=(str(ECHO_SERVER),),
        env={"FAKE_MCP_STATE_DIR": str(tmp_path), **env},
        startup_timeout_s=60.0,
        call_timeout_s=2.0,
    )


def read_attempts(tmp_path: Path) -> int:
    """读取替身记录的启动次数（没有文件时为 0）。"""

    counter = tmp_path / "startup_attempts.txt"
    return int(counter.read_text(encoding="utf-8")) if counter.is_file() else 0


def test_first_text_returns_first_text_block() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="hello")])
    assert first_text(result) == "hello"


def test_result_value_prefers_structured_content() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="ignored")],
        structuredContent={"result": True},
    )
    assert result_value(result) is True


def test_result_value_parses_json_text() -> None:
    result = CallToolResult(content=[TextContent(type="text", text='["a", "b"]')])
    assert result_value(result) == ["a", "b"]


def test_result_value_falls_back_to_raw_text() -> None:
    result = CallToolResult(content=[TextContent(type="text", text=r"C:\tmp\shot.png")])
    assert result_value(result) == r"C:\tmp\shot.png"


def test_result_value_without_content_is_none() -> None:
    assert result_value(CallToolResult(content=[])) is None


def test_round_trip_over_stdio_with_fake_server(tmp_path: Path) -> None:
    """启动离线 MCP server，走完 发现设备 -> 连接 -> 截图 -> 点击 -> 再截图。"""

    config = McpServerConfig(
        command=sys.executable,
        args=(str(FAKE_SERVER),),
        env={"FAKE_MCP_DIR": str(tmp_path)},
        startup_timeout_s=60.0,
        call_timeout_s=60.0,
    )

    async def flow() -> None:
        async with McpStdioClient(config) as client:
            tools = await client.list_tools()
            assert {"find_adb_device_list", "connect_adb_device", "screencap", "click"} <= set(
                tools
            )

            device = await MaaMcpDevice.connect(
                client, DeviceConfig(logical_width=1280, logical_height=720)
            )
            assert device.native_size == (1920, 1080)

            before = await device.screenshot()
            assert size_of(before) == (1280, 720)

            assert await device.tap(640, 360) is True
            after = await device.screenshot()
            assert diff_ratio(before, after) > 0.0

    asyncio.run(flow())

    clicks = [
        json.loads(line)
        for line in (tmp_path / "clicks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert clicks == [{"x": 960, "y": 540, "button": 0, "duration": 50}]


def test_call_text_type_mismatch_raises(tmp_path: Path) -> None:
    """工具返回布尔值但调用方要求字符串时必须报错，而不是静默转换。"""

    config = McpServerConfig(
        command=sys.executable,
        args=(str(FAKE_SERVER),),
        env={"FAKE_MCP_DIR": str(tmp_path)},
        startup_timeout_s=60.0,
        call_timeout_s=60.0,
    )

    async def flow() -> None:
        async with McpStdioClient(config) as client:
            with pytest.raises(Exception, match="期望返回字符串"):
                await client.call_text("click", {"controller_id": "ctrl_fake", "x": 1, "y": 1})

    asyncio.run(flow())


def test_handshake_survives_startup_noise(tmp_path: Path) -> None:
    """server 在握手前打印非 JSON 行时，会话仍能建立并调用工具。"""

    config = echo_server_config(tmp_path, FAKE_MCP_NOISE="startup")

    async def flow() -> tuple[list[str], str]:
        async with McpStdioClient(config) as client:
            tools = await client.list_tools()
            return tools, await client.call_text("echo", {"text": "ok"})

    tools, echoed = asyncio.run(flow())
    assert "echo" in tools
    assert echoed == "ok"


def test_call_survives_mid_session_noise(tmp_path: Path) -> None:
    """会话中途 stdout 混入非 JSON 行后，后续调用不失效。"""

    config = echo_server_config(tmp_path)

    async def flow() -> tuple[str, str]:
        async with McpStdioClient(config) as client:
            noisy = await client.call_text("noise")
            clean = await client.call_text("echo", {"text": "after-noise"})
            return noisy, clean

    assert asyncio.run(flow()) == ("ok", "after-noise")


def test_exit_is_clean_when_server_noises_on_shutdown(tmp_path: Path) -> None:
    """收尾噪声不能让 `async with` 抛异常（当前真实 MaaMCP 的行为）。"""

    config = echo_server_config(tmp_path, FAKE_MCP_NOISE="shutdown")

    async def flow() -> str:
        async with McpStdioClient(config) as client:
            return await client.call_text("echo", {"text": "before-shutdown"})

    assert asyncio.run(flow()) == "before-shutdown"


def test_call_times_out_with_mcp_tool_error(tmp_path: Path) -> None:
    """`sleep` 超过 call_timeout_s 时抛 McpToolError 且信息含工具名。"""

    config = echo_server_config(tmp_path)

    async def flow() -> None:
        async with McpStdioClient(config) as client:
            with pytest.raises(McpToolError, match="sleep"):
                await client.call("sleep", {"seconds": 5})

    asyncio.run(flow())


def test_startup_retries_with_backoff(tmp_path: Path) -> None:
    """前 N-1 次启动失败时按退避重试，第 N 次成功即返回。"""

    config = echo_server_config(tmp_path, FAKE_MCP_FAIL_TIMES="1").model_copy(
        update={"startup_retries": 2}
    )

    async def flow() -> str:
        async with McpStdioClient(config) as client:
            return await client.call_text("echo", {"text": "after-retry"})

    assert asyncio.run(flow()) == "after-retry"
    assert read_attempts(tmp_path) == 2


def test_startup_raises_after_exhausting_retries(tmp_path: Path) -> None:
    """重试用尽后抛 McpStartupError，并如实记录尝试次数。"""

    config = echo_server_config(tmp_path, FAKE_MCP_FAIL_TIMES="5").model_copy(
        update={"startup_retries": 1}
    )

    async def flow() -> None:
        with pytest.raises(McpStartupError, match="启动失败"):
            async with McpStdioClient(config):
                raise AssertionError("握手失败时不应进入会话")

    asyncio.run(flow())
    assert read_attempts(tmp_path) == 2


def test_tool_error_result_raises_mcp_tool_error(tmp_path: Path) -> None:
    """server 返回 isError 时抛 McpToolError 并带上 server 文本。"""

    config = echo_server_config(tmp_path)

    async def flow() -> None:
        async with McpStdioClient(config) as client:
            with pytest.raises(McpToolError, match="kaboom"):
                await client.call("boom", {"message": "kaboom"})

    asyncio.run(flow())


def test_session_property_raises_outside_context() -> None:
    """未进入 `async with` 时访问 session 要明确报错，而不是返回 None。"""

    client = McpStdioClient(McpServerConfig(command="unused"))
    with pytest.raises(RuntimeError, match="尚未建立"):
        _ = client.session

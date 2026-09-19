"""离线 MCP server 替身：复现真实 MaaMCP 的 stdout 噪声、卡死与启动失败。

真实 MaaMCP 在会话收尾（释放 ADB 控制器）时会把 ``product: MuMuPlayer-15.0-1``
这类纯文本打到 stdout，MCP 官方的 stdio_client 会把它当 JSON-RPC 消息解析并让
stdout_reader 任务崩掉。这个替身把同一行为做成可控开关，供离线测试复现。

环境变量：

- ``FAKE_MCP_NOISE``：``none`` / ``startup`` / ``shutdown``，决定噪声出现的位置；
  会话中途的噪声由 ``noise`` 工具主动触发。
- ``FAKE_MCP_FAIL_TIMES``：前 N 次启动直接退出（配合 ``FAKE_MCP_STATE_DIR`` 计数），
  用于验证客户端的启动退避重试。
- ``FAKE_MCP_STATE_DIR``：计数文件目录；``FAKE_MCP_FAIL_TIMES`` 必需。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

NOISE_ENV = "FAKE_MCP_NOISE"
FAIL_TIMES_ENV = "FAKE_MCP_FAIL_TIMES"
STATE_DIR_ENV = "FAKE_MCP_STATE_DIR"
NOISE_TEXT = "product: MuMuPlayer-15.0-1"
ATTEMPTS_FILE = "startup_attempts.txt"

mcp = FastMCP("FakeNoisyMCP")


def emit_noise(text: str = NOISE_TEXT) -> None:
    """向 stdout 打一行非 JSON 文本，模拟 MaaCore / adb 的噪声输出。"""

    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def startup_changed() -> bool:
    """若配置了 ``FAKE_MCP_FAIL_TIMES``：记录第几次启动并判断本次是否应失败。"""

    raw = os.environ.get(FAIL_TIMES_ENV)
    if not raw:
        return False
    state_dir = Path(os.environ[STATE_DIR_ENV])
    state_dir.mkdir(parents=True, exist_ok=True)
    counter = state_dir / ATTEMPTS_FILE
    attempts = int(counter.read_text(encoding="utf-8")) if counter.is_file() else 0
    attempts += 1
    counter.write_text(str(attempts), encoding="utf-8")
    return attempts <= int(raw)


@mcp.tool(name="echo")
def echo(text: str = "hello") -> str:
    """回显输入，用于确认会话在噪声之后仍然可用。"""

    return text


@mcp.tool(name="noise")
def noise(text: str = NOISE_TEXT) -> str:
    """会话中途打一行噪声到 stdout，然后正常返回。"""

    emit_noise(text)
    return "ok"


@mcp.tool(name="sleep")
def sleep(seconds: float = 1.0) -> str:
    """睡眠指定秒数，用于触发调用超时。"""

    time.sleep(seconds)
    return "slept"


@mcp.tool(name="boom")
def boom(message: str = "boom") -> str:
    """永远报错，用于验证 isError 分支。"""

    raise ValueError(message)


if __name__ == "__main__":
    noise_position = os.environ.get(NOISE_ENV, "none")
    if noise_position == "startup":
        emit_noise()
    if startup_changed():
        emit_noise("fake startup failure")
        raise SystemExit(3)
    mcp.run()
    if noise_position == "shutdown":
        emit_noise()

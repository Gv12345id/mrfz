"""基于 stdio 的最小 MCP 客户端。

职责边界：只负责启动 server、握手、调用工具、解析结果与超时控制，
不包含任何游戏业务逻辑（业务在 device/ 与 perception/ 层）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Mapping
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Self

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult

from arknights_agent.config import McpServerConfig
from arknights_agent.mcp.tools import first_text, result_value
from arknights_agent.mcp.transport import open_stdio_transport

logger = logging.getLogger(__name__)

# 启动重试的退避基数：第 n 次重试前等待 BASE * 2**(n-1) 秒。
STARTUP_BACKOFF_BASE_S = 0.5

# 收尾时只吞这些"连接断开"类异常：它们出现在会话已经确认结束之后，
# 不对应任何业务语义。其它异常照常抛出，避免把真实故障藏起来。
SHUTDOWN_NOISE_ERRORS = (
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)


class McpToolError(RuntimeError):
    """MCP 工具调用失败或返回错误。"""


class McpStartupError(RuntimeError):
    """MCP server 启动或握手失败（含重试耗尽）。"""


def _is_shutdown_noise(error: BaseException) -> bool:
    """判断异常（含异常组）是否只是收尾噪声。"""

    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(_is_shutdown_noise(item) for item in error.exceptions)
    return isinstance(error, SHUTDOWN_NOISE_ERRORS)


async def _close_quietly(stack: AsyncExitStack) -> None:
    """关闭未接管的资源栈；收尾阶段的断流异常不算失败。"""

    try:
        await stack.aclose()
    except BaseExceptionGroup as group:
        if not _is_shutdown_noise(group):
            raise
        logger.debug("重试前关闭会话时忽略断流异常：%r", group.exceptions)


class McpStdioClient:
    """启动 MCP server 子进程，并通过 stdio 调用其工具。"""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> Self:
        if shutil.which(self._config.command) is None:
            raise McpStartupError(
                f"找不到 MCP server 可执行文件 {self._config.command!r}："
                "请激活虚拟环境，或把它的目录加入 PATH"
            )
        attempts = self._config.startup_retries + 1
        last_error: BaseException | None = None
        for attempt in range(attempts):
            if attempt:
                delay = STARTUP_BACKOFF_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "MCP 会话第 %d/%d 次启动失败（%s: %s），%.1fs 后重试",
                    attempt,
                    attempts,
                    type(last_error).__name__,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                self._stack, self._session = await self._open()
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except BaseException as exc:
                last_error = exc
                continue
            return self
        raise McpStartupError(
            f"MCP 会话启动失败（已尝试 {attempts} 次）：{type(last_error).__name__}: {last_error}"
        ) from last_error

    async def _open(self) -> tuple[AsyncExitStack, ClientSession]:
        """开一次会话；失败时把已建立的资源关干净再抛。"""

        stack = AsyncExitStack()
        env = dict(os.environ)
        env.update(self._config.env)
        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=env,
        )
        try:
            read_stream, write_stream = await stack.enter_async_context(
                open_stdio_transport(params)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=self._config.startup_timeout_s)
        except BaseException:
            await _close_quietly(stack)
            raise
        return stack, session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is None:
            return
        try:
            await stack.aclose()
        except BaseExceptionGroup as group:
            if not _is_shutdown_noise(group):
                raise
            logger.warning("忽略 MCP 会话收尾时的断流异常：%r", group.exceptions)

    @property
    def session(self) -> ClientSession:
        """已建立的会话；未进入上下文时抛 RuntimeError。"""

        if self._session is None:
            raise RuntimeError("MCP 会话尚未建立，请使用 `async with McpStdioClient(...)`")
        return self._session

    async def list_tools(self) -> list[str]:
        """返回 server 暴露的工具名列表。"""

        listed = await asyncio.wait_for(
            self.session.list_tools(), timeout=self._config.call_timeout_s
        )
        return [tool.name for tool in listed.tools]

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> CallToolResult:
        """调用工具，返回原始结果；超时或 server 报错时抛 :class:`McpToolError`。"""

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(name, dict(arguments or {})),
                timeout=self._config.call_timeout_s,
            )
        except TimeoutError as exc:
            raise McpToolError(
                f"MCP 工具 {name} 调用超时（{self._config.call_timeout_s}s）"
            ) from exc
        if result.isError:
            detail = first_text(result) or "<无文本内容>"
            raise McpToolError(f"MCP 工具 {name} 返回错误：{detail}")
        return result

    async def call_value(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """调用工具并返回其实际返回值（见 :func:`mcp.tools.result_value`）。"""

        return result_value(await self.call(name, arguments))

    async def call_text(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """调用工具并要求返回字符串；返回其它类型时抛 :class:`McpToolError`。"""

        value = await self.call_value(name, arguments)
        if not isinstance(value, str):
            raise McpToolError(f"MCP 工具 {name} 期望返回字符串，实际为 {type(value).__name__}")
        return value

    async def call_bool(self, name: str, arguments: Mapping[str, Any] | None = None) -> bool:
        """调用工具并要求返回布尔值。"""

        value = await self.call_value(name, arguments)
        if not isinstance(value, bool):
            raise McpToolError(f"MCP 工具 {name} 期望返回布尔值，实际为 {type(value).__name__}")
        return value

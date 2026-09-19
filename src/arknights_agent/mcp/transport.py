"""stdio 传输：丢弃 MaaMCP 写在 stdout 的非 JSON 行。

背景（实测）：MaaMCP 在会话收尾释放 ADB 控制器时，会把
``product: MuMuPlayer-15.0-1`` 这类纯文本打到 stdout。MCP 官方
``mcp.client.stdio.stdio_client`` 会把每个换行分隔的片段当 JSON-RPC 消息
解析，解析失败就向 read stream 发送异常，进而让 ClientSession 的接收循环
崩掉、``async with`` 在退出时抛 ExceptionGroup。

这里照抄官方实现的进程管理与流结构，只改一件事：读到的行先判断是否长得像
JSON 对象，不是就记日志并跳过；解析失败的行同样只告警不杀任务组。传输出错
时上层表现为调用超时，而不是整段会话失效。

依赖 mcp 1.x 的内部辅助函数（``_create_platform_compatible_process`` /
``_terminate_process_tree``），所以 ``mcp`` 在 pyproject 里锁了 ``<2``；
升级 mcp 时必须重跑 ``tests/test_mcp_client.py`` 的噪声用例。
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TextIO

import anyio
import anyio.lowlevel
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import (
    StdioServerParameters,
    _create_platform_compatible_process,
    _terminate_process_tree,
    get_default_environment,
)
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

logger = logging.getLogger(__name__)

# JSON-RPC 消息一定以对象开头；其余行（adb / MaaCore 的输出）直接丢弃。
JSON_OBJECT_PREFIX = "{"

# 与官方实现一致的进程退出等待时间：超时后按 SIGTERM -> SIGKILL 升级。
PROCESS_TERMINATION_TIMEOUT_S = 2.0


async def _forward_line(line: str, writer: MemoryObjectSendStream[SessionMessage]) -> None:
    """把一行 stdout 转发为会话消息；不像 JSON-RPC 就丢弃。"""

    stripped = line.strip()
    if not stripped:
        return
    if not stripped.startswith(JSON_OBJECT_PREFIX):
        logger.debug("丢弃 stdout 上的非 JSON 行：%r", stripped[:200])
        return
    try:
        message = JSONRPCMessage.model_validate_json(stripped)
    except Exception as exc:  # 单行解析失败只告警，不让接收循环崩掉
        logger.warning("丢弃无法解析的 stdout 行（%s）：%r", exc, stripped[:200])
        return
    await writer.send(SessionMessage(message))


async def _stdout_reader(
    process: Process,
    writer: MemoryObjectSendStream[SessionMessage],
    params: StdioServerParameters,
) -> None:
    assert process.stdout, "进程缺少 stdout"
    try:
        async with writer:
            buffer = ""
            async for chunk in TextReceiveStream(
                process.stdout,
                encoding=params.encoding,
                errors=params.encoding_error_handler,
            ):
                # 一行可能跨多次 receive，必须按换行累积而不是"一次 receive 当一行"。
                lines = (buffer + chunk).split("\n")
                buffer = lines.pop()
                for line in lines:
                    await _forward_line(line, writer)
    except anyio.ClosedResourceError:
        await anyio.lowlevel.checkpoint()


async def _stdin_writer(
    process: Process,
    reader: MemoryObjectReceiveStream[SessionMessage],
    params: StdioServerParameters,
) -> None:
    assert process.stdin, "进程缺少 stdin"
    try:
        async with reader:
            async for session_message in reader:
                payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                await process.stdin.send(
                    (payload + "\n").encode(
                        encoding=params.encoding,
                        errors=params.encoding_error_handler,
                    )
                )
    except anyio.ClosedResourceError:
        await anyio.lowlevel.checkpoint()


@asynccontextmanager
async def open_stdio_transport(
    params: StdioServerParameters, errlog: TextIO = sys.stderr
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """启动 server 子进程并返回 (read, write) 两条内存流。"""

    read_writer: MemoryObjectSendStream[SessionMessage | Exception]
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_reader: MemoryObjectReceiveStream[SessionMessage]
    read_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_reader = anyio.create_memory_object_stream(0)

    try:
        process = await _create_platform_compatible_process(
            command=params.command,
            args=list(params.args),
            env=({**get_default_environment(), **(params.env or {})}),
            errlog=errlog,
            cwd=params.cwd,
        )
    except OSError:
        await read_stream.aclose()
        await write_stream.aclose()
        await read_writer.aclose()
        await write_reader.aclose()
        raise

    async with anyio.create_task_group() as task_group, process:
        task_group.start_soon(_stdout_reader, process, read_writer, params)
        task_group.start_soon(_stdin_writer, process, write_reader, params)
        try:
            yield read_stream, write_stream
        finally:
            # MCP spec 的 stdio 收尾顺序：先关 stdin，再等进程退出，最后收流。
            if process.stdin:
                with suppress(Exception):
                    await process.stdin.aclose()
            try:
                with anyio.fail_after(PROCESS_TERMINATION_TIMEOUT_S):
                    await process.wait()
            except TimeoutError:
                await _terminate_process_tree(process)
            except ProcessLookupError:
                pass
            await read_stream.aclose()
            await write_stream.aclose()
            await read_writer.aclose()
            await write_reader.aclose()

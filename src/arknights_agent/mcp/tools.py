"""MaaMCP 工具名常量与返回值解析。

工具名集中在这里，业务代码通过 :mod:`arknights_agent.mcp.tools` 引用，
避免工具名散落在各处。工具的真实 schema 以 MaaMCP 暴露的为准。
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolResult, TextContent

# 设备发现与连接
FIND_ADB_DEVICE_LIST = "find_adb_device_list"
CONNECT_ADB_DEVICE = "connect_adb_device"

# 屏幕与操作
SCREENCAP = "screencap"
CLICK = "click"
SWIPE = "swipe"
OCR = "ocr"


def first_text(result: CallToolResult) -> str | None:
    """返回结果中第一段文本内容，没有文本时返回 None。"""

    for block in result.content:
        if isinstance(block, TextContent):
            return block.text
    return None


def result_value(result: CallToolResult) -> Any:
    """取回工具的实际返回值。

    MaaMCP 通过 FastMCP 注册工具，返回值会出现在 ``structuredContent``
    的 ``result`` 字段；部分客户端只拿到文本块，因此文本再尝试按 JSON 解析，
    解析失败则按原样字符串返回。
    """

    structured: Any = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]

    text = first_text(result)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

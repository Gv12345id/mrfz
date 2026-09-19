"""设备层：截图与点击原语。"""

from arknights_agent.device.maa_via_mcp import DeviceError, MaaMcpDevice
from arknights_agent.device.protocol import DeviceBackend

__all__ = ["DeviceBackend", "DeviceError", "MaaMcpDevice"]

# MaaMCP 工具 schema 锁定（阶段 1）

> 本文是阶段 1 交付物：锁定 Agent 与 MAA 之间唯一通道上**实际在用**的工具名、参数字段与
> schema hash。工具名或 schema 变化时，本文与 `scripts/verify.py` 的 `schema-hash` check
> 会一起变红，必须重新查证后再放行（PLANS.md「resource 更新后必须重跑集成回归」）。

## 版本锁定

| 组件 | 版本 |
| --- | --- |
| `maa-mcp`（MaaMCP） | 1.2.3 |
| `MaaFw` | 5.13.1 |
| `mcp`（Python SDK） | 1.30.0 |
| Python | 3.12.10（`.venv`） |

## 结论：不需要在 adapter 侧补工具

实测 `tools/list` 返回 **24 个工具**，其中已经包含本项目所需的全部原语：

- 设备发现与连接：`find_adb_device_list`、`connect_adb_device`；
- 屏幕与坐标操作：`screencap`、`click`、`double_click`、`swipe`、`scroll`、`click_key`、`input_text`；
- 识别：`ocr`、`check_and_download_ocr`；
- Pipeline 生成与运行：`get_pipeline_protocol`、`save_pipeline`、`load_pipeline`、`run_pipeline`、`stop_pipeline`、`clear_pipeline_resources`、`benchmark_node`；
- 其他：`find_window_list`、`connect_window`、`save_captured_image`、`keyboard_shortcut`、`wait`、`get_current_datetime`。

因此 PLANS.md 阶段 1 任务 1 里「不提供时补最薄一层工具」的分支**不触发**，本轮没有对
MaaMCP 做任何本地补丁。

## 在用的 5 个工具与参数字段

下表按线上 `inputSchema` 逐字对齐（`required` 之外的字段都有默认值，可不传）。

| 工具 | 参数 | 类型 | 必填 | 默认 | 语义要点 |
| --- | --- | --- | --- | --- | --- |
| `find_adb_device_list` | — | — | — | — | 返回设备名列表；发现多台必须让用户选，不得自动挑 |
| `connect_adb_device` | `device_name` | string | ✅ | — | 返回 `controller_id`，后续所有调用都要带 |
| `screencap` | `controller_id` | string | ✅ | — | — |
| | `region` | `[int,int,int,int] \| null` | — | `null` | **按设备原生分辨率**理解；`null` 表示全屏 |
| | `resolution` | `int \| null` | — | `720` | **短边归一化目标**；1920×1080 → 1280×720；`null` 跳过归一化返回原生图 |
| `click` | `controller_id` | string | ✅ | — | — |
| | `x` / `y` | int | ✅ | — | **设备原生坐标**（本项目逻辑坐标 1280×720 经 `device/` 换算） |
| | `button` | int | — | `0` | ADB 下是手指编号 |
| | `duration` | int (ms) | — | `50` | 加大即长按 |
| `swipe` | `controller_id` | string | ✅ | — | — |
| | `start_x` / `start_y` / `end_x` / `end_y` | int | ✅ | — | 四个点都是**原生坐标** |
| | `duration` | int (ms) | ✅ | — | 无默认值，必须显式传 |

### 坐标语义（本项目的换算约定）

- 逻辑坐标系固定 **1280×720**；MaaMCP 的 `click` / `swipe` 收**原生坐标**，
  换算收口在 `src/arknights_agent/device/maa_via_mcp.py` 的 `to_native()`。
- 本机模拟器原生分辨率就是 1280×720，比例 1.0；非 16:9 设备由 `doctor` 直接判失败
  （`screencap` 得到的帧尺寸与逻辑坐标系不一致时 `DeviceError`）。

## schema hash

```
schema_sha256: 144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a
```

canonical 规则（`scripts/verify.py::canonical_schema_hash` 与本文一致）：

1. 取每个工具的 `{name, inputSchema}` 两个字段，**不含描述文本**（措辞会变，不该进 hash）；
2. 按工具名排序；
3. `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`；
4. 对 UTF-8 字节取 sha256（小写十六进制）。

### 复现命令

```powershell
# 直接复用门禁自己的实现，保证口径一致
$code = @'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("verify_script", Path("scripts/verify.py"))
verify = importlib.util.module_from_spec(spec)
sys.modules["verify_script"] = verify
spec.loader.exec_module(verify)
tools = verify.live_schema_tools()
print(len(tools), verify.canonical_schema_hash(tools))
'@
$code | .\.venv\Scripts\python.exe -
# 期望：24 144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a
```

### 门禁比对

```powershell
.\.venv\Scripts\python.exe scripts\verify.py --phase 1 --only schema-doc,schema-hash
# schema-doc: 文档存在且含工具名、inputSchema 参数表与 sha256
# schema-hash: 现场重算 hash 与本文一致
```

## 实测行为（2026-09-19，本机 MuMu）

| 观测 | 值 |
| --- | --- |
| `find_adb_device_list` | `["MuMu安卓设备-1-MuMuPlayer v5+"]`（单台） |
| `screencap(resolution=720)` | 稳定返回 **1280×720** PNG，单帧 40–64ms，100 帧 p95 **63.751ms** |
| `click` 行为 | 逻辑 (992,335)「干员」入口 ↔ 逻辑 (50,40)「返回」各 10 次，20/20 画面变化确认 |
| 收尾噪声 | MaaMCP 释放 ADB 控制器时向 stdout 打印 `product: MuMuPlayer-15.0-1`（非 JSON），需由 `src/arknights_agent/mcp/transport.py` 过滤 |

## 升级流程（改这里也要改门禁）

1. 升级 `maa-mcp` / `MaaFw` 后先跑本节「复现命令」，记录新 hash；
2. 更新本文的版本表、在用的工具参数表与 `schema_sha256`；
3. 跑 `pytest -m "not integration"`（离线回放）与 `pytest -m integration tests/integration/test_device_loop.py`（真机回归）；
4. 跑 `python scripts\verify.py --phase 1`，确认 `schema-doc` / `schema-hash` 与真机项同时绿。

# 阶段 1（环境搭建 · MCP 调通 MAA）任务级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Python 侧通过 MCP 稳定控制 MAA：拿到 1280×720 截图、把点击打到指定逻辑坐标，并用 `scripts/verify.py --phase 1` 单入口判定放行。

**Architecture:** `Codex 决策层 → MCP(stdio) → maa-mcp-adapter → MaaCore → 模拟器`。桥接层只做 schema 校验与会话转发；设备层把「逻辑坐标 1280×720」与「设备原生坐标」的换算收口在 `device/`，上层只依赖 `DeviceBackend` Protocol。门禁把每条验收标准映射成一个可比对的 check，真机类 check 在拿不到设备时如实 `passed=false`。

**Tech Stack:** Python 3.11+（本机 3.12.10）、mcp 1.x（已锁 `<2`）、maa-mcp 1.2.3、MaaFw 5.13.1、OpenCV、pydantic、typer、pytest / pytest-cov / ruff / mypy。

**Spec:** `AGENTS.md`（架构、规范、红线）、`PLANS.md`（阶段 1 交付物与验收标准、全局基础设施 JSON 契约）、`docs/specs/2026-09-19-mcp-tool-schema.md`（本阶段产出，MCP 工具名与参数的事实来源）。

## Global Constraints

以下约束来自 `AGENTS.md` / `PLANS.md`，每个任务都隐含遵守，不再逐条重复：

- Python 3.11+，`src/` 布局，包名 `arknights_agent`；PEP 8，行长 100，格式化交给 `ruff format`。
- 逻辑坐标系固定 1280×720；设备非 16:9 时 `doctor` 直接失败退出。
- 决策层与 MAA 之间只走 MCP；`src/arknights_agent/` 内不出现 `adb`、窗口句柄或第三方设备库调用。
- 公共函数、方法、类属性写全量类型注解（含返回值），用 `X | None`，不用 `Optional[X]`；层间数据用 pydantic 或 `@dataclass(frozen=True)`；跨层接口用 `Protocol`。
- 每个等待都有超时与退避；`asyncio` 中不阻塞事件循环（`asyncio.to_thread`）。
- 日志走 `logging.getLogger(__name__)`，路径走 `pathlib.Path`。
- 默认测试全程离线（`pytest -m "not integration"`），真机测试标 `@pytest.mark.integration`。
- 模型权重、数据集、运行日志、`runs/`、`artifacts/` 不入库；本机私有信息只进 `config/config.local.yaml`。
- 提交信息中文，格式 `type: 描述`，一次提交只做一件事；提交前 `ruff format --check` / `ruff check` / `mypy src` / `pytest` 全绿。
- 阶段放行判据只有一个：`python scripts/verify.py --phase 1` 退出码 0 且 JSON 里所有 `checks[].passed == true`。**真机类 check 不允许用 skip 或放宽阈值换绿。**

## 2026-09-19 现场实测（本计划的事实基础）

计划的前提必须来自观测，不是假设。下面每条都是本次实测结果：

| 观测 | 证据 |
| --- | --- |
| MaaMCP 可独立启动，握手 1.6s，暴露 **24 个工具**，含 `screencap` / `click` / `swipe` / `ocr` | 见"验证证据" |
| **本机存在可用设备**：`find_adb_device_list` 返回 `MuMu安卓设备-1-MuMuPlayer v5+`，`connect_adb_device` 返回 controller_id | 同上 |
| `screencap(resolution=720)` 稳定返回 **1280×720** PNG，10 帧实测单帧 53.6–57.1ms | 同上 |
| 模拟器里**已装明日方舟且当前停在战斗界面**（费用 30、建设值 0/20、6 名干员待部署） | `view_image` 人眼确认 |
| MaaMCP 在**会话收尾时**向 stdout 打印非 JSON 行 `product: MuMuPlayer-15.0-1`，导致 MCP `stdout_reader` 解析失败 → 退出上下文时抛 `ExceptionGroup` | 同上 |
| 该噪声**只出现在收尾**：连接后 sleep 5s 再连拍 3 帧全部成功，异常只在离开 `async with` 时出现 | 同上 |
| MaaMCP 截图落在 `%LOCALAPPDATA%\MaaXYZ\MaaMCP\screenshots\`，**会话结束即清理**，所以"先截图后读文件"必须在会话内完成 | 同上 |

结论：**阶段 1 的真机门禁不是"缺设备"阻塞，而是两个工程问题**——

1. MaaMCP 收尾噪声会让 `McpStdioClient.__aexit__` 抛异常，导致 CLI 与集成测试退出码非 0（Task 1 修）；
2. 游戏当前停在战斗界面，20 次点击闭环必须先由人把游戏恢复到中立界面再跑，否则点击会打在关卡里（Task 7 的安全闸门 + Task 8 的人工前置）。

### 验证证据（复现命令与输出）

```powershell
# 1) 工具清单与 schema hash（自建 stdio 会话，不经过仓库代码）
$code | .\.venv\Scripts\python.exe -
# 输出：handshake_s=1.6 / tools=24 / find_adb_device_list … benchmark_node
#      schema_sha256= 144f391f06492bdffa88859b39058d7b40bd93b8b12cbb99a7ecb84eb766c01a

# 2) 真机截图 10 帧
# 输出：STEP connect ok 'e61788a5-…' / STEP cap 0..9 (1280, 720, 53.6~57.1ms)
#      Failed to parse JSONRPC message from server … input_value='product: MuMuPlayer-15.0-1\r'
#      ABORTED: ExceptionGroup unhandled errors in a TaskGroup (1 sub-exception)

# 3) 收尾噪声定位
# 输出：connect ok … / slept 5s / cap after sleep 0..2 isError= False
#      → 噪声只在收尾阶段出现
```

## 门禁到 check 的映射（`scripts/verify.py --phase 1`）

| PLANS.md 阶段 1 验收标准 | check 名 | `value` | `threshold` |
| --- | --- | --- | --- |
| 1. `ruff format --check` / `ruff check` / `mypy src` 全绿 | `static-checks` | `null` | `null` |
| 2. `pytest -m "not integration"` 全绿 | `offline-tests` | 通过用例数 | `null` |
| 2. `mcp/` 覆盖率 ≥ 80% | `coverage-mcp` | 覆盖率 % | `80.0` |
| 2. `device/` 覆盖率 ≥ 80% | `coverage-device` | 覆盖率 % | `80.0` |
| 6. schema 文档存在且含工具名/参数表/hash | `schema-doc` | `null` | `null` |
| 6. 文档 hash 与线上 MaaMCP schema 一致 | `schema-hash` | `null`（hash 是字符串，按契约 `value`/`threshold` 只放数字，比对结果写进 `detail`） | `null` |
| 4. `arknights-agent doctor --json` 退出码 0 且字段达标 | `doctor-json` | `null` | `null` |
| 3. `pytest -m integration tests/integration/test_device_loop.py` 全绿 | `device-loop-tests` | `null` | `null` |
| 3. 100 帧截图 0 失败 | `screenshot-failures` | 失败帧数 | `0` |
| 5. `screenshot_p95_ms ≤ 1500` | `screenshot-p95-ms` | p95 毫秒 | `1500` |
| 3. 20 次点击闭环 20/20 | `click-loop-successes` | 成功次数 | `20` |

`passed` = 所有 check 的逻辑与；任一失败退出码非 0；`value` / `threshold` 一律是机器可比对的数字（不适用时为 `null`）。

## 文件结构

| 文件 | 动作 | 职责 |
| --- | --- | --- |
| `src/arknights_agent/mcp/client.py` | 改 | 自建 stdout 过滤传输 + 启动退避重试 + 收尾容错 |
| `src/arknights_agent/mcp/transport.py` | 新建 | stdio 传输：丢弃非 JSON 行，其余交给 `ClientSession` |
| `src/arknights_agent/device/protocol.py` | 改 | 补齐 `swipe()` / `close()`（PLANS 固定签名） |
| `src/arknights_agent/device/maa_via_mcp.py` | 改 | 实现 `swipe()` / `close()`，坐标换算复用 `to_native` |
| `src/arknights_agent/device/fake.py` | 改 | `FakeDevice` 对齐同一接口 |
| `src/arknights_agent/cli.py` | 改 | 全局 `--config-dir`；`doctor --json --frames 100` |
| `src/arknights_agent/__main__.py` | 新建 | 支持 `python -m arknights_agent`（CI 与 verify.py 的稳定入口） |
| `src/arknights_agent/config.py` | 改 | `McpServerConfig.startup_retries`；`integration` 段（安全确认 + 点击目标） |
| `scripts/verify.py` | 新建 | 阶段门禁入口（本阶段唯一放行判据） |
| `tests/fixtures/mcp_echo_server.py` | 新建 | 离线 MCP server：echo / sleep / fail / 注入 stdout 噪声 |
| `tests/test_mcp_client.py` | 改 | 握手、超时、重试、收尾噪声、错误返回的离线用例 |
| `tests/test_device.py` | 改 | `swipe`/`close` 与坐标换算用例 |
| `tests/test_cli_doctor.py` | 新建 | `doctor --json` 字段与退出码（用假 server） |
| `tests/test_verify_contract.py` | 新建 | verify.py 的 JSON 契约与聚合逻辑（纯函数） |
| `tests/integration/test_device_loop.py` | 新建 | 真机：100 帧 + 20 次点击闭环 + metrics 落盘 |
| `docs/specs/2026-09-19-mcp-tool-schema.md` | 新建 | 锁定 24 个工具、5 个在用工具的参数字段与 schema hash |
| `.gitignore` | 改 | 补 `runs/`、`artifacts/` |
| `pyproject.toml` | 改 | dev 依赖加 `pytest-cov>=5.0` |

---

## Task 1: MCP stdio 会话的抗噪声与重试

**Files:**
- Create: `src/arknights_agent/mcp/transport.py`
- Modify: `src/arknights_agent/mcp/client.py`、`src/arknights_agent/config.py`
- Create: `tests/fixtures/mcp_echo_server.py`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Consumes: `McpServerConfig(command, args, env, startup_timeout_s, call_timeout_s)`。
- Produces:
  - `async def open_stdio_transport(params: StdioServerParameters) -> tuple[MemoryObjectReceiveStream, MemoryObjectSendStream]`（`mcp.transport`）
  - `McpStdioClient(config: McpServerConfig)`：`async with` 可用；`list_tools() -> list[str]`；`call(name, arguments=None) -> CallToolResult`；`call_value/call_text/call_bool`；失败抛 `McpToolError`。
  - `McpServerConfig.startup_retries: int = 2`（握手失败后按 0.5s / 1s 退避重试）。

- [ ] **Step 1: 写失败的离线夹具与测试**

`tests/fixtures/mcp_echo_server.py`（关键部分，行为对齐真实 MaaMCP 的收尾噪声）：

```python
FAKE_MCP_NOISE = "FAKE_MCP_NOISE"  # none | startup | shutdown | mid
FAKE_MCP_DELAY_S = "FAKE_MCP_DELAY_S"


@mcp.tool(name="noise")
def noise(text: str = "product: MuMuPlayer-15.0-1") -> str:
    """向 stdout 打一行非 JSON 文本，模拟 MaaCore/adb 的噪声输出。"""

    sys.stdout.write(text + "\n")
    sys.stdout.flush()
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
    if os.environ.get(FAKE_MCP_NOISE) == "startup":
        sys.stdout.write("noise: startup\n")
        sys.stdout.flush()
    mcp.run()
    if os.environ.get(FAKE_MCP_NOISE) == "shutdown":
        sys.stdout.write("noise: shutdown\n")
        sys.stdout.flush()
```

`tests/test_mcp_client.py` 新增用例：

```python
def test_handshake_survives_startup_noise(tmp_path: Path) -> None:
    """server 在握手前打印非 JSON 行时，会话仍能建立并调用工具。"""


def test_call_survives_mid_session_noise(tmp_path: Path) -> None:
    """会话中途 stdout 混入非 JSON 行后，后续调用不失效。"""


def test_exit_is_clean_when_server_noises_on_shutdown(tmp_path: Path) -> None:
    """收尾噪声不能让 `async with` 抛异常（当前真实 MaaMCP 的行为）。"""


def test_call_times_out_with_mcp_tool_error(tmp_path: Path) -> None:
    """`sleep` 超过 call_timeout_s 时抛 McpToolError 且信息含工具名。"""


def test_startup_retries_with_backoff(tmp_path: Path) -> None:
    """前 N-1 次启动失败时按退避重试，第 N 次成功即返回。"""


def test_tool_error_result_raises_mcp_tool_error(tmp_path: Path) -> None:
    """server 返回 isError 时抛 McpToolError 并带上 server 文本。"""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_client.py -q`
Expected: 新增用例 FAIL（`ModuleNotFoundError: arknights_agent.mcp.transport` / 收尾抛 `ExceptionGroup` / 噪声后调用失效）

- [ ] **Step 3: 实现过滤传输**

`src/arknights_agent/mcp/transport.py`：

```python
"""stdio 传输：丢弃 MaaMCP 写在 stdout 的非 JSON 行。

MaaMCP 在收尾（释放 ADB 控制器）时会把 `product: MuMuPlayer-15.0-1` 这类
文本打到 stdout，MCP 官方 stdio_client 会把它当 JSON-RPC 消息解析并让
stdout_reader 任务崩掉。这里在读到换行后先判断该行是否是 JSON 对象，
不是就丢进日志、继续读下一行。
"""

_JSONRPC_PREFIX = b"{"


async def _stdout_reader(process: Process, read_stream_writer: ...) -> None:
    while True:
        line = await process.stdout.receive()
        if not line:
            return
        if not line.lstrip().startswith(_JSONRPC_PREFIX):
            logger.debug("丢弃 MaaMCP 的非 JSON stdout 行：%r", line[:200])
            continue
        await read_stream_writer.send(...)
```

要点（照抄 `mcp.client.stdio` 的形状，只加过滤）：

1. 保留 `anyio` 的 memory object stream + `create_task_group` 结构，`SessionMessage` 解析失败时只在**该行**上告警而不是杀任务组。
2. `process.stdout` 的缓冲区按 ``\n`` 切分；一行可能跨多次 `receive()`，必须用 `TextIOWrapper`/`readline()` 而不是"每次 receive 当一行"。
3. `errlog` 转发 stderr（MaaMCP 的日志走 stderr，别丢）。

- [ ] **Step 4: 收尾容错 + 启动退避重试**

`McpStdioClient.__aenter__`：握手失败时按 `0.5 * 2**attempt` 退避重试，最多 `startup_retries + 1` 次；每次重试前把上一次的 `AsyncExitStack` 关干净。

`McpStdioClient.__aexit__`：关闭传输时若抛 `BaseExceptionGroup`，只在**全部子异常都是收尾噪声导致的解析/断流错误**时吞掉并 `logger.warning`，其它异常照常抛出。

- [ ] **Step 5: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_mcp_client.py -q`
Expected: PASS

- [ ] **Step 6: 真机复验收尾不再抛异常**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import asyncio; from arknights_agent.cli import main" ; .\.venv\Scripts\arknights-agent.exe doctor
```

Expected: 退出码 0（Task 3 完成后 doctor 才有 `--json`，此处先看退出码）

- [ ] **Step 7: 提交**

```bash
git add src/arknights_agent/mcp tests/fixtures/mcp_echo_server.py tests/test_mcp_client.py src/arknights_agent/config.py
git commit -m "fix: MCP stdio 会话忽略 MaaMCP 非 JSON 输出并支持启动重试"
```

## Task 2: 设备层补齐 PLANS 固定接口（`swipe` / `close`）

**Files:**
- Modify: `src/arknights_agent/device/protocol.py`、`src/arknights_agent/device/maa_via_mcp.py`、`src/arknights_agent/device/fake.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: `McpStdioClient.call_bool`；MaaMCP `swipe(controller_id, start_x, start_y, end_x, end_y, duration) -> bool`（实测 inputSchema 的字段名与顺序）。
- Produces:
  - `DeviceBackend.swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool`（逻辑坐标）
  - `DeviceBackend.close() -> None`（幂等；关闭后任何调用抛 `DeviceError`）

- [ ] **Step 1: 写失败测试**

```python
def test_swipe_converts_all_four_coordinates_to_native() -> None:
    """(10, 20) -> (100, 200) 在 1920x1080 上应下发 (15, 30) -> (150, 300)。"""


def test_close_is_idempotent_and_blocks_later_calls() -> None:
    """close() 两次不报错；之后再 tap 抛 DeviceError。"""


def test_fake_device_records_swipes_and_closes() -> None:
    """FakeDevice.swipe 记录逻辑坐标，close 后 taps/swipes 不再增长。"""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_device.py -q`
Expected: FAIL（`AttributeError: 'MaaMcpDevice' object has no attribute 'swipe'`）

- [ ] **Step 3: 实现**

```python
async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> bool:
    """在逻辑坐标之间滑动；MaaMCP 只认原生坐标，四个点都要换算。"""

    self._ensure_open()
    native_x1, native_y1 = self.to_native(x1, y1)
    native_x2, native_y2 = self.to_native(x2, y2)
    return await self._client.call_bool(
        SWIPE,
        {
            "controller_id": self._controller_id,
            "start_x": native_x1,
            "start_y": native_y1,
            "end_x": native_x2,
            "end_y": native_y2,
            "duration": duration_ms,
        },
    )


def close(self) -> None:
    """幂等关闭。

    MaaMCP 没有"断开设备"工具（实测 24 个工具里只有 connect_*，也没有
    disconnect_*），因此这里只做本地状态收口：标记关闭、拒绝后续调用。
    真正的资源释放由 MaaMCP 进程退出承担（`McpStdioClient` 退出上下文）。
    """

    self._closed = True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_device.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/arknights_agent/device tests/test_device.py
git commit -m "feat: 设备层补齐 swipe/close 原语"
```

## Task 3: CLI 的 `doctor --json` 与 `python -m arknights_agent`

**Files:**
- Create: `src/arknights_agent/__main__.py`
- Modify: `src/arknights_agent/cli.py`、`src/arknights_agent/config.py`
- Test: `tests/test_cli_doctor.py`

**Interfaces:**
- Consumes: `load_settings(config_dir) -> Settings`、`MaaMcpDevice.connect`、`McpStdioClient`。
- Produces:
  - `arknights-agent --config-dir <dir> doctor --json --frames 100` → stdout 只有一段 JSON：`{mcp, maa, device, resolution, screenshot_p95_ms, frames, device_name, controller_id, native_size, tools}`；全 ok 退出码 0，任一失败退出码 1。
  - `python -m arknights_agent ...` 等价于 `arknights-agent ...`。
  - `IntegrationConfig(confirm_screen: bool = False, tap_a: tuple[int, int] | None, tap_b: tuple[int, int] | None)`（供 Task 7 使用，落在 `config.local.yaml`）。

- [ ] **Step 1: 写失败测试**

```python
def test_doctor_json_reports_ok_fields_with_fake_server(tmp_path: Path) -> None:
    """用假 MCP server 跑 doctor --json，字段齐全且退出码 0。"""


def test_doctor_exits_nonzero_when_no_device(tmp_path: Path) -> None:
    """假 server 返回空设备列表时 device=fail、退出码 1。"""


def test_doctor_reports_screenshot_p95_ms(tmp_path: Path) -> None:
    """--frames 5 时 JSON 里 screenshot_p95_ms 是数字且 frames == 5。"""


def test_p95_helper_is_nearest_rank_percentile() -> None:
    """p95([...]) 用最近秩法，5 个样本时取最大值。"""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_doctor.py -q`
Expected: FAIL（`No such option: --json`）

- [ ] **Step 3: 实现**

```python
def p95_ms(samples_ms: Sequence[float]) -> float:
    """最近秩法 p95（样本升序后取 ceil(0.95 * n) 位），样本为空抛 ValueError。"""


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="只输出 JSON")] = False,
    frames: Annotated[int, typer.Option(help="测延迟用的截图帧数")] = 100,
) -> None:
    report = asyncio.run(_doctor(frames))
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    raise typer.Exit(code=0 if report["mcp"] == report["maa"] == report["device"] == "ok" else 1)
```

`--config-dir` 用 typer 的 `@app.callback()` 全局选项实现，写进 `ctx.obj`，`load_settings` 接收该目录（`config.local.yaml` 因此可以本机私有）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_doctor.py -q`
Expected: PASS

- [ ] **Step 5: 真机确认**

Run: `.\.venv\Scripts\python.exe -m arknights_agent doctor --json --frames 100`
Expected: 退出码 0，`resolution=1280x720`，`screenshot_p95_ms` 远低于 1500

- [ ] **Step 6: 提交**

```bash
git add src/arknights_agent/cli.py src/arknights_agent/__main__.py src/arknights_agent/config.py tests/test_cli_doctor.py
git commit -m "feat: doctor 支持 --json 与 100 帧截图延迟统计"
```

## Task 4: MCP 工具 schema 查证与锁定文档

**Files:**
- Create: `docs/specs/2026-09-19-mcp-tool-schema.md`

**Interfaces:**
- Consumes: 线上 MaaMCP（`maa-mcp 1.2.3` + `MaaFw 5.13.1`）的 `tools/list` 结果。
- Produces: 文档中的 schema hash（canonical 规则：按工具名排序，取 `{name, inputSchema}`，`json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)` 的 sha256），供 `verify.py` 的 `schema-hash` check 比对。

- [ ] **Step 1: 落盘工具清单与 hash**

文档必须包含：

1. 24 个工具名的完整清单（证据：一次 `tools/list`）。
2. 在用的 5 个工具的参数表：`find_adb_device_list()`、`connect_adb_device(device_name)`、`screencap(controller_id, region?, resolution?)`、`click(controller_id, x, y, button?, duration?)`、`swipe(controller_id, start_x, start_y, end_x, end_y, duration)`，并注明 `screencap.resolution` 是**短边归一化**、`click`/`swipe` 收**原生坐标**。
3. 版本锁定：`maa-mcp 1.2.3`、`MaaFw 5.13.1`、`mcp 1.30.0`。
4. `schema_sha256` 与复现命令（一段可直接粘贴的 `python -c`/heredoc 脚本）。
5. 本地补丁决策：**不需要补 adapter 工具**（截图与坐标点击原语齐全），因此 PLANS.md 风险项"是否暴露截图原语与坐标点击原语"记为已确认。

- [ ] **Step 2: 校验文档与线上一致**

Run: `.\.venv\Scripts\python.exe scripts\verify.py --phase 1 --only schema-doc,schema-hash`
Expected: 两个 check `passed=true`

- [ ] **Step 3: 提交**

```bash
git add docs/specs/2026-09-19-mcp-tool-schema.md
git commit -m "docs: 锁定 MaaMCP 工具 schema 与 hash"
```

## Task 5: `scripts/verify.py` 门禁入口

**Files:**
- Create: `scripts/verify.py`
- Create: `tests/test_verify_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 仓库根目录、venv 解释器（`sys.executable`）、`runs/phase1/device_loop.json`（Task 7 产出）。
- Produces:
  - `python scripts/verify.py --phase 1 [--only a,b] [--json-out PATH]` → stdout 一段 JSON（PLANS.md 契约），退出码 0/1；阶段 2–4 未实现时退出码 2 并说明。
  - `CheckResult(name, passed, detail="", value=None, threshold=None)`、`build_report(phase, checks, started_at) -> VerifyReport`、`exit_code(report) -> int`、`aggregate_coverage(coverage_json, package) -> float`、`canonical_schema_hash(tools) -> str`。

- [x] **Step 1: 写契约测试**（本轮完成，证据见文末）

```python
def test_passed_is_logical_and_of_checks() -> None: ...
def test_exit_code_is_nonzero_when_any_check_fails() -> None: ...
def test_report_matches_plans_json_contract() -> None: ...
def test_aggregate_coverage_sums_covered_over_statements() -> None: ...
def test_canonical_schema_hash_changes_when_schema_changes() -> None: ...
```

- [x] **Step 2: 实现**（本轮完成）

`scripts/verify.py` 用 `subprocess.run(..., timeout=...)` 跑：`ruff format --check .`、`ruff check .`、`mypy src`、`pytest -m "not integration" --cov=arknights_agent.mcp --cov=arknights_agent.device --cov-report=json:`、`pytest -m integration tests/integration/test_device_loop.py`、`python -m arknights_agent doctor --json`，再算 schema hash 与读 metrics JSON。所有真机类 check 在无设备时 `passed=false` 且 `detail` 写明原因，**不 skip、不放宽 threshold**。

- [x] **Step 3: 跑通与自检**（本轮完成）

Run: `.\.venv\Scripts\python.exe scripts\verify.py --phase 1 --json-out runs\verify\phase-1.json`
Expected: 输出契约 JSON；未完成的交付物对应 check 为 `false`（离线的 static/offline-tests 应为 `true`）

- [ ] **Step 4: 提交**

```bash
git add scripts/verify.py tests/test_verify_contract.py .gitignore
git commit -m "feat: 新增阶段门禁入口 scripts/verify.py"
```

## Task 6: 覆盖率门禁（`mcp/` 与 `device/` ≥ 80%）

**状态：** 已完成（2026-09-19）。实测 `mcp/` 90.72%、`device/` 87.68%，都在 80% 之上，
不需要补测试；依赖与阈值已接进 verify.py，后续掉下 80% 会直接变红。

**Files:**
- Modify: `pyproject.toml`（dev 依赖加 `pytest-cov>=5.0`，已改）
- Test: `tests/test_mcp_client.py`、`tests/test_device.py`、`tests/test_cli_doctor.py`（本轮未新增，留作掉线时的补测清单）

**Interfaces:**
- Consumes: Task 1/2/3 的全部实现。
- Produces: `pytest -m "not integration" --cov=arknights_agent.mcp --cov=arknights_agent.device --cov-report=json:runs/coverage.json` 的 `totals`。

- [x] **Step 1: 装依赖并测基线**

```powershell
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "pytest-cov>=5.0"
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q --cov=arknights_agent.mcp --cov=arknights_agent.device --cov-report=term-missing
```

输出：`Successfully installed coverage-7.16.1 pytest-cov-7.1.0`，
`coverage-mcp value=90.72`、`coverage-device value=87.68`。

- [x] **Step 2: 补覆盖不足的分支 —— 本轮无缺口，无需补测**

下面这张表是**覆盖率掉到 80% 以下时的补测清单**，本轮没用到：

| 位置 | 用例 |
| --- | --- |
| `mcp/client.py` 超时/错误/会话未建立 | `test_call_times_out_with_mcp_tool_error`、`test_tool_error_result_raises_mcp_tool_error`、`test_session_property_raises_outside_context` |
| `mcp/client.py` 启动重试 | `test_startup_retries_with_backoff` |
| `device/maa_via_mcp.py` connect 失败与非 PNG 返回 | `test_connect_raises_when_controller_missing`、`test_screenshot_rejects_non_png_path` |
| `device/maa_via_mcp.py` 原生分辨率测量 | `test_measure_native_size_uses_unscaled_screencap` |
| `device/fake.py` swipe/close | `test_fake_device_records_swipes_and_closes` |

- [x] **Step 3: 确认达标**

Run: `.\.venv\Scripts\python.exe scripts\verify.py --phase 1 --only coverage-mcp,coverage-device`
实际：两个 check `passed=true`（`value=90.72` / `value=87.68`，`threshold=80`）

- [ ] **Step 4: 提交**（未提交：工作区仍有大量未入库文件，等阶段 1 收官一起提交）

```bash
git add pyproject.toml tests
git commit -m "test: 补齐 mcp/device 覆盖率至 80% 以上"
```

## Task 7: 真机集成测试（100 帧 + 20 次点击闭环）

**Files:**
- Create: `tests/integration/__init__.py`、`tests/integration/test_device_loop.py`

**Interfaces:**
- Consumes: `McpStdioClient`、`MaaMcpDevice`、`load_settings()`、`diff_ratio`、`IntegrationConfig(confirm_screen, tap_a, tap_b)`。
- Produces: `runs/phase1/device_loop.json`：

```json
{"frames": 100, "frame_failures": 0, "p95_ms": 57.1, "clicks_attempted": 20,
 "clicks_confirmed": 20, "device": "MuMu安卓设备-1-MuMuPlayer v5+", "resolution": "1280x720"}
```

- [ ] **Step 1: 写测试（含安全闸门）**

```python
pytestmark = pytest.mark.integration


def test_hundred_frames_and_twenty_click_loop() -> None:
    """100 帧截图 0 失败；20 次"点击已知元素 → 画面变化"闭环全部确认。"""
```

安全闸门（写死在测试里，不靠提示词）：

1. `config.local.yaml` 里 `integration.confirm_screen: true` 才允许点击；否则 `pytest.fail("未确认游戏处于中立界面…")`。
2. `integration.tap_a` / `tap_b` 必须配置；点击序列为 `A, B` 交替 10 轮（A 展开面板、B 收起），每轮两次点击都断言 `diff_ratio(before, after) > 0.02`。
3. 点击前先取一帧作为基线；若 100 帧阶段任何一帧尺寸 ≠ 1280×720，直接失败（非 16:9 不换算、不猜测）。
4. metrics JSON 在 `finally` 里落盘，失败也写，避免 verify.py 读到陈旧文件。

- [ ] **Step 2: 离线自检（不碰真机）**

Run: `.\.venv\Scripts\python.exe -m pytest -m "not integration" -q`
Expected: PASS（真机测试被 marker 排除）

- [ ] **Step 3: 真机跑（人工前置：把游戏恢复到中立界面）**

Run: `.\.venv\Scripts\python.exe -m pytest -m integration tests/integration/test_device_loop.py -q`
Expected: PASS，且 `runs/phase1/device_loop.json` 中 `frame_failures=0`、`clicks_confirmed=20`、`p95_ms ≤ 1500`

- [ ] **Step 4: 提交**

```bash
git add tests/integration
git commit -m "test: 新增真机 100 帧与 20 次点击闭环集成测试"
```

## Task 8: 阶段 1 放行

**Files:** 无新增（只跑门禁与提交）

- [ ] **Step 1: 全量门禁**

Run: `.\.venv\Scripts\python.exe scripts\verify.py --phase 1 --json-out runs\verify\phase-1.json`
Expected: 退出码 0，JSON 中 `passed=true`，11 个 check 全 `true`

- [ ] **Step 2: 人工抽查**

确认 `runs/verify/phase-1.json` 中 `screenshot-p95-ms.value ≤ 1500`、`click-loop-successes.value = 20`、`screenshot-failures.value = 0`，并与 PLANS.md 的验收标准逐条对齐。

- [ ] **Step 3: 更新 PLANS.md 阶段进度**

把 `阶段进度` 里 `- [ ] 阶段 1` 改成 `- [x] 阶段 1`，附上 `runs/verify/phase-1.json` 的 `started_at`。

```bash
git add PLANS.md
git commit -m "docs: 标记阶段 1 门禁通过"
```

---

## 自查（Self-Review）

**1. Spec 覆盖**

| PLANS.md 阶段 1 要求 | 落点 |
| --- | --- |
| 任务 1 查证 MCP 工具清单、决定是否需要补 adapter 工具 | Task 4（实测已确认原语齐全，无需补丁） |
| 任务 2 `config.py` + `mcp/client.py` 与离线 echo server 单测 | 已有 + Task 1（噪声/超时/重试） |
| 任务 3 `device/` 三件套 + 非 16:9 拒绝 | 已有 + Task 2（`swipe`/`close`） |
| 任务 4 `doctor` / `capture` / `tap` | Task 3（`--json`、`--frames`、退出码） |
| 任务 5 `integration/test_device_loop.py` | Task 7 |
| 验收 1 静态检查 | `static-checks` |
| 验收 2 离线测试 + 覆盖率 | `offline-tests`、`coverage-mcp`、`coverage-device`、Task 6 |
| 验收 3 100 帧 / 20 次点击 | `device-loop-tests`、`screenshot-failures`、`click-loop-successes`、Task 7 |
| 验收 4 `doctor --json` | `doctor-json`、Task 3 |
| 验收 5 `screenshot_p95_ms ≤ 1500` | `screenshot-p95-ms` |
| 验收 6 schema 文档 | `schema-doc`、`schema-hash`、Task 4 |
| 全局基础设施 `scripts/verify.py` | Task 5 |

**2. 占位符扫描**：无 TBD / 无"稍后补"，每个 check 都有名字、阈值与数据来源。

**3. 前后一致性**：`swipe(x1, y1, x2, y2, duration_ms)` 在 Task 2 定义、Task 7 未使用（本阶段点击闭环只用 `tap`）；`screenshot_p95_ms` 在 Task 3 产出、`screenshot-p95-ms` check 在 Task 5 读取；`IntegrationConfig` 字段名 `confirm_screen` / `tap_a` / `tap_b` 在 Task 3 定义、Task 7 使用，命名一致。

## 已知风险与处理

- **收尾噪声是 MaaMCP 的既有行为**：Task 1 用过滤 + 收尾容错处理；若未来 MaaMCP 版本修掉，测试保持正向（噪声存在时也要工作），不会因此变红。
- **点击闭环会碰真机**：Task 7 的闸门保证"没有人工确认就不点"，且失败时先落 metrics 再退出。
- **schema hash 会随 MaaMCP 升级变化**：升级时 Task 4 的文档与 hash 必须同步更新，`schema-hash` check 会红，这是期望行为（PLANS.md「resource 更新后必须重跑集成回归」）。

## 执行记录

### 2026-09-19 · Task 5（`scripts/verify.py`）已完成

新增 `scripts/verify.py`、`tests/test_verify_contract.py`；`.gitignore` 补 `runs/`、`artifacts/`；
`pyproject.toml` dev 依赖补 `pytest-cov>=5.0`（Task 6）。

```powershell
> .\.venv\Scripts\python.exe -m pytest tests\test_verify_contract.py -q
11 passed in 0.05s

> .\.venv\Scripts\python.exe -m ruff format --check . ; .\.venv\Scripts\python.exe -m ruff check .
26 files already formatted
All checks passed!

> .\.venv\Scripts\python.exe -m mypy src
Success: no issues found in 11 source files

> .\.venv\Scripts\python.exe -m pytest -m "not integration" -q
37 passed in 2.76s

> .\.venv\Scripts\python.exe scripts\verify.py --phase 1
exit_code=1
阶段 1 门禁：未通过｜static-checks=ok offline-tests=ok coverage-mcp=ok coverage-device=ok
  schema-doc=FAIL schema-hash=FAIL doctor-json=FAIL device-loop-tests=FAIL
  screenshot-failures=FAIL screenshot-p95-ms=FAIL click-loop-successes=FAIL
  - schema-doc: 缺少 docs\specs\2026-09-19-mcp-tool-schema.md
  - schema-hash: 缺少 docs\specs\2026-09-19-mcp-tool-schema.md，无法比对 hash
  - doctor-json: doctor 退出码 1：No module named arknights_agent.__main__
  - device-loop-tests: 缺少 tests\integration\test_device_loop.py
  - screenshot-failures / screenshot-p95-ms / click-loop-successes:
    缺少 runs\phase1\device_loop.json：真机集成测试尚未成功产出指标

> checks 数值一览（来自 runs\verify\phase-1.json）
static-checks       passed=True   value=None    threshold=None
offline-tests       passed=True   value=37      threshold=None
coverage-mcp        passed=True   value=90.72   threshold=80
coverage-device     passed=True   value=87.68   threshold=80
schema-doc          passed=False  value=None    threshold=None
schema-hash         passed=False  value=None    threshold=None
doctor-json         passed=False  value=None    threshold=None
device-loop-tests   passed=False  value=None    threshold=None
screenshot-failures passed=False  value=None    threshold=0
screenshot-p95-ms   passed=False  value=None    threshold=1500
click-loop-successes passed=False value=None    threshold=20
```

红的原因全部是"交付物还没做"（Task 1–4、7），不是设备缺失：4 个离线 check 已经真绿。
按顺序执行后，`doctor-json` 会因 Task 3 新增的 `__main__.py` 而通过，真机三项 check 会在
Task 7 产出 `runs/phase1/device_loop.json` 后取到 `value`。

### 待办（按顺序）

1. Task 1 MCP stdio 抗噪声（当前真机会在收尾抛 `ExceptionGroup`，必须最先修）。
2. Task 3 `doctor --json`（另外补 `__main__.py`，verify.py 依赖它）。
3. Task 4 schema 文档（含 hash `144f391f…c01a`，已实测）。
4. Task 7 真机集成测试（需人工先把游戏从战斗界面退回中立界面）。
5. Task 2 `swipe`/`close`（阶段 2/3 才用，可排在真机门禁之后）。

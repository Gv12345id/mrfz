# 明日方舟代肝脚本 — 设计文档

- 日期：2026-09-10
- 状态：已确认（待实施计划）
- 项目路径：`E:\Codex\arknights-auto`

## 1. 背景与目标

构建一个自研的明日方舟自动化脚本，在 Android 模拟器上完成日常重复操作（俗称"代肝"）。

第一版目标范围：

1. **刷体力**：自动选择关卡、开始作战、处理结算、循环直到理智耗尽后停机。
2. **日常杂项**：剿灭作战、每日签到、领取邮件与任务奖励、公开招募。

成功标准：在无人值守的情况下连续运行到理智耗尽，全程无需人工干预，且失败时能安全停机而不是乱点。

## 2. 非目标（v1 明确不做）

- 不做图形界面（GUI），以命令行 + 配置文件驱动。
- 不做多开、多账号并发（但架构上保留扩展位）。
- 不做肉鸽、保全派驻、基建换班（状态机复杂度远高于上述目标，后续单独评估）。
- 不内置、不下载、不分发任何游戏美术资源。所有模板均从用户本机截图裁剪。

## 3. 约束与前提

- 运行环境：Android 模拟器（首选 MuMu12），通过 ADB 通信。
- 逻辑坐标系固定为 **1280×720 横屏**。运行时不接受非 16:9 分辨率，直接报错退出。
- 语言与版本：Python 3.11+。
- 单实例串行运行，同一时刻只操作一台设备。
- 本工具使用第三方自动化方式操作游戏，违反游戏用户协议。风险由使用者自行承担。

## 4. 总体架构

```
CLI / 配置 (config.yaml)
        ↓
调度层 Scheduler    任务队列、优先级、失败重试、理智耗尽则停机
        ↓
任务层 Tasks        刷体力 / 剿灭 / 公招 / 签到 / 领邮件（各一个状态机）
        ↓
动作层 Actions      tap_element / wait_page / expect_page / swipe（内置等待与重试）
        ↓
感知层 Vision       页面分类 + 模板匹配 + OCR + 颜色检测
        ↓
IO 层               CaptureBackend(截图) / InputBackend(点击)
```

### 核心约束：单向数据流

任务层与动作层**不允许**直接调用 ADB、不允许自行处理图像，只能使用动作层提供的语义接口。

这条约束带来两个必要的能力：

- **离线回放**：把 IO 层替换为"读磁盘图片 + 记录点击"，整套任务逻辑可在无游戏环境下运行与测试。
- **后端可替换**：ADB 换为模拟器厂商的高速截图 API 时，任务层无需改动。

## 5. 组件设计

### 5.1 IO 层

```python
@dataclass(frozen=True)
class Frame:
    image: np.ndarray          # BGR，已缩放至 1280x720 逻辑坐标系
    raw_size: tuple[int, int]  # 设备原始分辨率
    ts: float

class CaptureBackend(Protocol):
    def capture(self) -> Frame: ...
    def close(self) -> None: ...

class InputBackend(Protocol):
    def tap(self, x: int, y: int) -> None: ...
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None: ...
```

- `AdbCaptureBackend`：`adb exec-out screencap -p`，解码 PNG 后缩放到 1280×720。
- `AdbInputBackend`：`adb shell input tap / swipe`。
- 坐标在后端内部完成从逻辑坐标系到设备坐标系的换算，上层只见 1280×720。

v1 直接使用 `subprocess` 调用 `adb`，不引入 `adbutils` 等额外抽象。若后续设备管理复杂度上升再评估。

### 5.2 感知层

感知层是全项目风险最集中的部分，90% 的失败模式都源于此。

```python
@dataclass(frozen=True)
class Match:
    element: str
    bbox: tuple[int, int, int, int]
    score: float

@dataclass(frozen=True)
class PageResult:
    page: str | None
    score: float
    runner_up: str | None
    certain: bool

class Vision:
    def find(self, element: str, frame: Frame) -> Match | None: ...
    def find_all(self, element: str, frame: Frame) -> list[Match]: ...
    def page(self, frame: Frame) -> PageResult: ...
    def ocr_int(self, element: str, frame: Frame) -> int | None: ...
```

**页面分类器。** 每个页面定义 3–5 个锚点（小模板，可选颜色块），**全部在固定 ROI 内匹配**。固定 ROI 单次匹配耗时 1–3ms，全屏匹配则需数十毫秒，且误匹配率显著更高。分类器对所有页面打分，取最高分与次高分；最高分过阈值**且与次高分拉开足够差距**时判定为 `certain=True`。

**"不确定就不动"原则。** `certain=False` 时任务层不得发出点击，只能等待下一帧或进入兜底流程。乱点一次可能把账号带到十步之外的位置，恢复成本远高于多等 200ms。这是整个脚本最重要的可靠性设计。

**模板匹配。** OpenCV `matchTemplate`，`TM_CCOEFF_NORMED`，阈值按元素配置。模板在启动时按实际分辨率缩放并缓存，避免每帧重算。

**OCR。** 仅用于数字与短文本（理智值、招募 tag、任务进度）。采用 PaddleOCR（PP-OCRv4，ONNX Runtime，CPU）。单次耗时 50–150ms，因此**只在需要的页面调用，绝不每帧执行**。关键数值（理智）需连续两次读数一致方可采信。

**分辨率策略。** 启动时检测设备分辨率；非 16:9 直接拒绝运行。所有元素 ROI 以 1280×720 逻辑坐标声明，运行时按比例换算。

### 5.3 动作层 / Context

第 4 节中的「动作层」即由 `Context` 实现。它是任务层可见的唯一接口：

```python
class Context:
    def page(self) -> str | None: ...
    def wait_page(self, page: str, *, timeout: float = 10.0) -> Frame: ...
    def expect_page(self, page: str) -> None: ...
    def tap(self, element: str, *, timeout: float = 3.0) -> None: ...
    def tap_xy(self, x: int, y: int) -> None: ...
    def swipe(self, x1, y1, x2, y2, duration_ms: int) -> None: ...
    def read_int(self, element: str) -> int | None: ...
    def sleep(self, seconds: float) -> None: ...
    def log(self, msg: str) -> None: ...
```

`tap(element)` 的语义是「找到元素 → 点击 → 确认状态已改变」：找不到元素则按退避策略重试，超时抛 `ActionError`。动作层内部同样遵守"不确定就不动"原则。

### 5.4 任务层

任务为**阻塞式**函数，内部循环推进状态，通过 `ctx` 原语与游戏交互：

```python
class Task(Protocol):
    name: str
    def run(self, ctx: Context) -> TaskResult: ...
```

阻塞式而非生成器/协程式，是为了让流程可以按线性代码阅读、打断点、看堆栈。代价是调度器无法在动作中途插入控制，因此全局停机条件由 `ctx` 在每个原语调用处检查，命中即抛出 `StopRequested` 向上冒出并终止任务。

**刷体力流程：**

```
主界面 → 点终端 → 关卡选择 → 选关卡 → 开始行动 → 确认
     → 战斗中（等"作战结束"特征）
     → 结算（跳过 → 确认）→ 回到关卡选择 ↺
     → 理智不足 → 结束任务
```

**剿灭 / 公招 / 签到 / 领邮件** 各自独立实现，共享动作层原语。

每个动作后跟一次 `expect_page(...)` 校验，超时抛异常交由调度层处理。

### 5.5 调度层

- 按 `config.yaml` 中的顺序执行任务，支持任务级启用开关与优先级。
- 任务失败时按配置重试（默认 2 次），仍失败则记录并切换到下一个任务。
- 全局停机条件：理智耗尽、连续失败达上限、ADB 连接丢失、用户中断。
- 所有任务结束后输出一次运行摘要（各任务执行轮次、耗时、失败原因）。

### 5.6 CLI

基于 `typer`：

```
arknights-auto doctor                       # 检查 ADB 连接、分辨率、截图耗时
arknights-auto capture --out assets/captures/
arknights-auto crop --frame <png> --rect x,y,w,h --name <element>
arknights-auto run [--task combat] [--dry-run]
```

`--dry-run` 连接真机但不发出任何点击，仅打印「识别到 X，准备点 (x,y)」。接入新界面时依赖此模式。

## 6. 资源与配置

### elements.yaml

声明元素名到 ROI、阈值、处理方式的映射。坐标为 1280×720 逻辑坐标。

```yaml
elements:
  terminal.entry:
    template: templates/terminal_entry.png
    roi: [980, 560, 280, 140]
    threshold: 0.85
  sanity.value:
    kind: ocr_int
    roi: [40, 20, 160, 40]
    scale: 3
```

### pages.yaml

声明页面到锚点集合的映射。

```yaml
pages:
  main:
    anchors: [main.terminal_entry, main.sanity_icon]
  level_select:
    anchors: [level_select.start_button]
```

### config.yaml

任务开关、关卡选择、重试次数、ADB 设备地址、超时参数。

### 素材标注工具

`tools/annotate.py`：打开一帧截图 → 鼠标框选 → 输入元素名 → 自动裁剪存至 `assets/templates/` 并写入 `elements.yaml`。使用 Python 标准库 tkinter，不引入额外依赖。由于模板必须来自用户本机截图，此工具是必需品而非可选项。

## 7. 错误处理与恢复

| 情况 | 处理 |
|---|---|
| 页面不确定（`certain=False`） | 等待下一帧，最多 N 帧；仍不确定则进入恢复流程 |
| 元素找不到 | 退避重试；超时抛 `ActionError` |
| 卡在未知界面/弹窗 | 执行 `recover_to_home()`：循环尝试返回键与关闭弹窗，直到识别到主界面 |
| 任务失败 | 调度层重试；超过上限则跳过并记录 |
| ADB 断开 | 立即中止运行并输出明确错误 |

`recover_to_home()` 是长时间无人值守运行的关键路径，必须独立于具体任务实现，并有对应测试。

## 8. 测试策略

核心是**离线回放**，否则每次验证都要开游戏，迭代速度无法接受。

1. `tests/fixtures/` 存放一批真实截图及 `labels.yaml` 标注的期望页面。
2. 最高价值测试：页面分类器对每张 fixture 返回正确页面。界面改版时它最先报警。
3. 元素定位测试：验证各元素在 fixture 上的命中与不命中符合预期。
4. 任务状态机测试：用 `FakeIO` 喂入固定帧序列，断言产生的动作序列。
5. 全部测试使用 `pytest`，不需要模拟器与网络。

## 9. 目录结构

```
E:\Codex\arknights-auto\
├─ pyproject.toml
├─ README.md
├─ config/config.yaml
├─ docs/specs/
├─ src/arknights_auto/
│  ├─ io/            base.py, adb.py, fake.py
│  ├─ vision/        page.py, match.py, ocr.py
│  ├─ actions.py
│  ├─ context.py
│  ├─ tasks/         base.py, combat.py, annihilation.py, recruit.py, daily.py
│  ├─ scheduler.py
│  └─ cli.py
├─ assets/
│  ├─ templates/     从本机截图裁剪（不入库）
│  ├─ captures/
│  ├─ elements.yaml
│  └─ pages.yaml
├─ tools/annotate.py
└─ tests/
   ├─ fixtures/
   └─ test_*.py
```

## 10. 依赖

`opencv-python`、`numpy`、`onnxruntime`、`paddleocr`、`pyyaml`、`typer`、`pytest`（开发）。ADB 通过 `subprocess` 直调。Python 3.11+。

## 11. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M0 地基 | ADB 截图/点击后端、`capture` 与 `doctor` 命令 | 在模拟器上稳定取得 1280×720 帧；截图耗时与坐标换算可测量、点击落点准确 |
| M1 感知 | 页面分类器、模板匹配、标注工具 | 能稳定识别主界面、终端、关卡选择；fixture 回归测试通过 |
| M2 刷体力闭环 | 完整作战状态机 | 无人干预循环至理智耗尽并安全停机 |
| M3 日常杂项 | 剿灭、签到、领邮件、公招 | 各任务独立可运行，全部通过运行摘要正确汇报 |
| M4 稳定性 | 异常恢复、离线回放测试补全 | 从任意弹窗状态可恢复到主界面；fixture 覆盖全部已支持页面 |

建议 M0 与 M1 连续完成，先交付一个可验证的切片：抓帧 → 识别主界面 → 点进终端。

## 12. 风险与未决问题

**风险**

- 模拟器/真机实际分辨率与 DPI 差异导致坐标偏移（用 16:9 硬校验 + `doctor` 诊断缓解）。
- 游戏版本更新改变 UI，导致模板失效（用 fixture 回归测试尽早发现，标注工具让重做成本低）。
- 低对比度小字 OCR 不稳定（关键数值需双次读数一致）。
- 界面切换动画期间截图导致误判（依赖"不确定就不动"，必要时增加稳定帧确认）。
- 使用第三方自动化违反游戏用户协议，存在账号风险。

**未决问题（可在实施中决定）**

- 具体模拟器品牌与是否启用厂商高速截图 API。v1 先走通用 ADB 路径，`doctor` 输出实测截图耗时后决定是否优化。
- 关卡选择采用固定位置点击还是滑动查找。视关卡列表是否固定而定。
- 是否需要自动开启"代理指挥"与二倍速，还是一次性手动配置好。
- 理智耗尽后的策略：直接停机，还是使用理智合剂后继续。

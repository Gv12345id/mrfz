# AGENTS.md — 明日方舟自我进化 AI Agent

本仓库构建一个能在《明日方舟》中自我进化的 AI Agent：**Codex 决策、MAA 操作、MCP 通信**。本文件是常驻指令；四阶段实施计划与各阶段验收门禁见 `PLANS.md`。`docs/specs/2026-09-10-arknights-auto-design.md` 与 `docs/plans/2026-09-11-arknights-core-loop.md` 描述的是早期自研 ADB + OpenCV 方案，只在需要历史背景时翻阅，目标架构以本文件为准。

## 1. 项目目标

无人值守跑完日常（刷体力、剿灭作战、公开招募、签到、领邮件），失败时安全停机而不是乱点；每个动作都能回溯到当时的观察、理由与结果；策略靠评测对比迭代变强。

范围之外：多账号并发、GUI、游戏客户端注入/封包/内存读写（见「红线」）。

## 2. 架构概述

```
Codex 决策层 → MCP 工具调用 → maa-mcp-adapter → MAA Python 接口 → MaaCore + resource → ADB → 模拟器
```

一轮循环：读状态 → 决策 → 下发 MAA 任务 → 收结构化结果 → 写运行记录 → 评测 → 更新策略。

三个支点：

- **唯一通道**：决策层与 MAA 之间只走 MCP。任务编排、超时、重试、停机判断都在决策层；桥接层只做 schema 校验与会话转发。
- **确定性执行**：MAA 负责识别、点击与内置任务流程；Agent 负责选任务、判异常、决定何时停。
- **评测门禁**：策略或提示词的改动，都要给出基线 vs 候选的评测对比。

早期方案里「不确定就不动」的原则同样适用：状态不明确时等下一帧，不做猜测性点击。

## 3. 技术栈与布局

Python 3.11+、`src/` 布局；PyTorch 承载可学习组件（默认 CPU，模型缺失时退回规则策略）；MAA Python 接口；MCP（stdio）；pydantic / pyyaml / typer；pytest / ruff / mypy。

```
src/arknights_agent/
  mcp/       MCP 客户端、会话与工具调用封装
  device/    截图/点击原语（基于 MCP 的实现 + 离线 Fake）
  perception/ YOLOv8 检测、OCR、状态向量
  policy/    决策层：多头 DQN、动作掩码、训练与检查点
  memory/    运行记录、经验回放、数据集构建
  evolution/ Analyzer → Coder → 自动验证循环
  eval/      离线评测与基线回归
  config.py  cli.py
config/      config.yaml（入库）、config.local.yaml（本机私有）
tests/       fixtures/ 放录制的 MCP 会话与状态快照
docs/        specs/ 与 plans/，命名 YYYY-MM-DD-<主题>.md
```

新模块按层归位，不平铺在根目录。

## 4. 编码规范

- PEP 8，行长 100。格式与静态检查交给工具，不手工对齐：

```
ruff format .    ruff check .    mypy src    pytest
```

- 公共函数、方法、类属性写全量类型注解（含返回值）；用 `X | None`，不用 `Optional[X]`。
- 层间数据用 pydantic 模型或 `@dataclass(frozen=True)`；裸 `dict` 留在边界解析处。
- 跨层接口用 `Protocol` 声明，方便 mock 与回放。
- 每个等待都有超时与退避；`asyncio` 中不阻塞事件循环（用 `asyncio.to_thread`）。
- 日志走 `logging.getLogger(__name__)`，路径走 `pathlib.Path`。

## 5. 测试

pytest，默认全程离线：**默认不碰真机、不碰游戏**。

- 决策与桥接逻辑用 `tests/fixtures/` 的录制会话回放：同一份输入序列必须得到同一份决策序列。
- 真机/模拟器测试标 `@pytest.mark.integration`，用 `pytest -m "not integration"` 整体排除。
- 测试文件镜像被测模块路径（`policy/planner.py` → `tests/test_planner.py`）；函数名 `test_<行为>_<条件>_<期望>`。
- 验收标准：策略类改动附基线 vs 候选的评测数字；MAA 交互类改动在真机回归一次并附日志摘要。

## 6. 关键外部依赖

**maa-mcp-adapter** — Agent 与 MAA 之间的唯一通道。锁版本；工具名与参数以它暴露的 schema 为准，先拉 schema 再写调用；升级时先跑回放测试，再跑集成测试。需要本地补丁时在 `docs/` 记录原因与合并计划。

**MaaAssistantArknights** — MaaCore 与 resource 成对锁定。resource 更新会改变任务名与行为，升级后必须重跑集成回归。只走官方 Python 接口。许可为 AGPL-3.0，对外分发前确认义务。

**PyTorch** — 默认 CPU 版，CUDA 走可选依赖组。权重、数据集、运行日志不入库。

依赖统一写在 `pyproject.toml` 并锁定；改完本地环境不等于改完依赖。

## 7. 红线

- 只用 MAA 公开能力操作游戏；注入、封包篡改、内存读写、反检测规避都不属于本项目。
- 凭证、设备地址等本机信息只进 `config/config.local.yaml`；游戏素材、模型权重、运行日志不入库。
- 无人值守运行必须有停机条件：理智耗尽、连续失败上限、连接丢失、用户中断。

## 8. 协作与交付

- 提交信息用中文，格式 `type: 描述`，type 取 feat / fix / test / docs / chore / refactor。
- 提交前 `ruff format --check`、`ruff check`、`mypy src`、`pytest` 全绿；一次提交只做一件事。
- 接入新界面或新任务时，先 `arknights-agent run --dry-run`（连真机但不发出点击）确认决策输出。

## 9. 常用命令

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
arknights-agent doctor
```

架构或目录发生实质变化时，同步更新本文件。

# 明日方舟自我进化 Agent · 四阶段实施计划

> **For agentic workers:** 本文件是阶段级计划（phase-gate plan）。开始某一阶段前，先用 `superpowers:writing-plans` 为该阶段生成任务级计划，落到 `docs/plans/YYYY-MM-DD-phaseN-<主题>.md`，再用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 执行。任务级计划里每一步用 `- [ ]` 勾选跟踪。

**Goal:** 交付一个能在《明日方舟》0-1 关卡自主作战、并通过 Analyzer → Coder → 验证的闭环自我进化的 Agent。

**Architecture:** Codex 决策、MAA 操作、MCP 通信。感知层（YOLOv8 + OCR）把战场截图转成固定维度的状态向量；决策层（多头 DQN + 动作掩码）输出动作；进化层复用评测集与真机回归组成自动闭环。每阶段的放行由 `scripts/verify.py --phase N` 单一入口判定。

**Tech Stack:** Python 3.11+、PyTorch、Ultralytics YOLOv8、MAA Python 接口、MCP（stdio）、OpenCV、pydantic、pytest / ruff / mypy。

**Spec:** `AGENTS.md`（架构与规范）、`docs/specs/2026-09-10-arknights-auto-design.md`（历史方案，参考其"不确定就不动"与离线回放原则）。

**关联文档:** `AGENTS.md` 定义编码规范与红线，本文件不重复；两者冲突时以 `AGENTS.md` 为准。

## 阶段进度

- [x] 阶段 1：环境搭建 —— MCP 调通 MAA，拿到截图、点得准（2026-09-19 门禁通过：`scripts/verify.py --phase 1` 退出码 0，11/11 全绿，详见 `runs/phase1-acceptance.md`）
- [ ] 阶段 2：感知模块 —— YOLOv8 + OCR → 结构化状态向量
      （分步进度 2026-09-19：步骤 1 完成（100 帧只读采集 + 标注流程验证，采集器已升级 v2）；
      步骤 2 完成（两批合并去重后 426 帧干净战斗帧，`runs/phase2/annotation_400.json`）；
      步骤 3 第一轮训练完成但**未达标**：mAP@0.5=0.4879（目标 0.85），原因见验收记录；
      数据已扩充到 996 帧战斗帧并重新预标注（1581 框），等人工抽查后重训；
      步骤 4 费用 OCR 完成（准确率 100%），CD 进行中。详见 `runs/phase2-acceptance.md`）
- [ ] 阶段 3：决策模块 —— 多头 DQN + 动作掩码，0-1 关卡可训练
- [ ] 阶段 4：自主进化循环 —— Analyzer → Coder → 自动重跑验证

## Global Constraints

以下约束适用于每个阶段，任务级计划不再重复。

- Python 3.11+，`src/` 布局，包名 `arknights_agent`。
- 逻辑坐标系固定 **1280×720**；设备非 16:9 时 `doctor` 直接失败退出。
- 决策层与 MAA 之间**只走 MCP**；`src/arknights_agent/` 内的游戏交互全部经由 `device/` 与 `mcp/`。
- 单帧状态向量维度固定 **173**（见阶段 2 定义），感知层之外不再出现图像类型。
- 默认测试全程离线（`pytest -m "not integration"`）；真机/模拟器测试标 `@pytest.mark.integration`。
- 模型权重、数据集、运行日志、`runs/`、`artifacts/` 一律不入库（`.gitignore`）。
- 提交信息中文，格式 `type: 描述`；一次提交只做一件事。
- 每个阶段的唯一放行判据：`python scripts/verify.py --phase N` 退出码为 0，且生成的 JSON 中所有 `checks[].passed == true`。

## 门禁总览

| 阶段 | 交付物 | 停止条件（全部满足才放行） |
| --- | --- | --- |
| 1 | MCP 会话 + 设备原语 + `doctor` | 静态检查全绿；离线测试全绿；真机 100 帧截图 0 失败；20 次点击闭环 20/20；截图 p95 ≤ 1500ms |
| 2 | 感知模块 + 状态向量 | 敌人 mAP@50 ≥ 0.75；敌人存在性召回 ≥ 0.90；费用/CD 整数值准确率 ≥ 0.98；状态向量维度 = 173；单帧 p95 ≤ 300ms |
| 3 | 多头 DQN + 掩码 + 训练脚本 | 掩码不变量测试通过；离线回报 ≥ 随机基线 × 1.3；seed 复现差异 ≤ 1%；0-1 真机 20 局通关 ≥ 16 局 |
| 4 | Analyzer / Coder / 沙箱 / 循环 | 白名单拦截 20/20；坏补丁 0 keep 且回滚干净；评测集 hash 不变；5 轮内 ≥1 keep；通关率 ≥ 80% 且不低于基线 |

## 全局基础设施（阶段 1 内完成，后续阶段复用）

`scripts/verify.py` 是所有停止条件的执行入口，契约固定：

```json
{
  "phase": 1,
  "passed": true,
  "started_at": "2026-09-19T10:00:00+08:00",
  "checks": [
    {"name": "static-checks", "passed": true, "detail": "ruff/format/check/mypy ok", "value": null, "threshold": null},
    {"name": "screenshot-p95-ms", "passed": true, "detail": "", "value": 412.0, "threshold": 1500}
  ]
}
```

规则：`passed` 为所有 `checks[].passed` 的逻辑与；任一 check 失败则退出码非 0；`value` / `threshold` 用于把阈值写成机器可比对的数字而不是文字。CI 与阶段 4 的循环都只读这个 JSON。

---

## 阶段 1：环境搭建

**目标：** Python 侧通过 MCP 调用 MAA，稳定拿到 1280×720 截图，并能把点击打到指定逻辑坐标上。

**交付物（文件与职责）：**

| 文件 | 职责 |
| --- | --- |
| `src/arknights_agent/config.py` | pydantic 配置模型：MCP server 启动命令、设备串口、超时、私有配置加载（`config/config.yaml` + `config/config.local.yaml`） |
| `src/arknights_agent/mcp/client.py` | MCP stdio 会话：启动/握手/调用/超时/重连；`call_tool(name, args) -> dict` |
| `src/arknights_agent/mcp/tools.py` | 从会话拉取工具清单并缓存为强类型 wrapper；工具名集中在此，禁止散落调用 |
| `src/arknights_agent/device/protocol.py` | `DeviceBackend` Protocol：`screenshot()` / `tap(x, y)` / `swipe(...)` / `close()` |
| `src/arknights_agent/device/maa_via_mcp.py` | 基于 MCP 的 `DeviceBackend` 实现，负责像素尺寸校验与坐标换算 |
| `src/arknights_agent/device/fake.py` | 离线 `FakeDevice`：读磁盘 PNG 队列 + 记录点击，供全部离线测试使用 |
| `src/arknights_agent/cli.py` | `doctor` / `capture` / `tap` 三个命令 |
| `scripts/verify.py` | 门禁入口（见上节契约） |
| `docs/specs/2026-09-19-mcp-tool-schema.md` | 查证并锁定所用 MCP 工具名、参数、返回结构与 schema hash |
| `tests/` | `test_config.py`、`test_mcp_client.py`、`test_device_coords.py`、`integration/test_device_loop.py` |

**关键接口（后续阶段依赖，签名固定）：**

```python
class DeviceBackend(Protocol):
    def screenshot(self) -> np.ndarray: ...  # BGR, 已归一化到 1280x720
    def tap(self, x: int, y: int) -> None: ...  # 逻辑坐标
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None: ...
    def close(self) -> None: ...
```

**任务：**

1. 查证 MCP adapter 暴露的实际工具清单：跑一次会话、dump 工具 schema、回答"是否提供截图原语与坐标点击原语"。不提供时，在 adapter 侧补最薄的一层工具（只做截图/点击/查询状态），并在 `docs/specs/2026-09-19-mcp-tool-schema.md` 记录决策与理由。
2. 实现 `config.py` + `mcp/client.py`，离线单测用本地 echo MCP server（`tests/fixtures/mcp_echo_server.py`）验证握手、超时、重连。
3. 实现 `device/` 三件套，坐标换算单测覆盖 16:9 与非 16:9（后者必须抛错拒绝运行）。
4. 实现 `cli.py` 的 `doctor`（输出 JSON：mcp / maa / device / resolution / screenshot_p95_ms）、`capture --out`、`tap --xy`。
5. 写 `integration/test_device_loop.py`：连续 100 帧截图 + 20 次"点击已知元素 → 截图确认状态变化"。

**验收标准（可自动验证）：**

1. `ruff format --check .`、`ruff check .`、`mypy src` 全绿。
2. `pytest -m "not integration"` 全绿；`src/arknights_agent/mcp` 与 `src/arknights_agent/device` 覆盖率 ≥ 80%。
3. `pytest -m integration tests/integration/test_device_loop.py` 全绿：100 帧截图 0 失败、20/20 点击闭环成功。
4. `arknights-agent doctor --json` 退出码 0，字段满足 `mcp=ok`、`maa=ok`、`device=ok`、`resolution=1280x720`。
5. `screenshot_p95_ms ≤ 1500`（100 帧实测，写进 doctor JSON）。
6. `docs/specs/2026-09-19-mcp-tool-schema.md` 存在，含所用工具名、参数表与 schema hash。

**停止条件：** `python scripts/verify.py --phase 1` 退出码 0。

---

## 阶段 2：感知模块

**目标：** 把战场截图转成固定维度的结构化状态向量，供决策层直接消费。

**交付物（文件与职责）：**

| 文件 | 职责 |
| --- | --- |
| `src/arknights_agent/perception/detector.py` | Ultralytics YOLOv8 推理封装：`detect(frame) -> list[EnemyBox]`；模型路径与置信度阈值来自配置 |
| `src/arknights_agent/perception/ocr.py` | 费用 / 干员 CD 数字识别；双次读数一致才采信 |
| `src/arknights_agent/perception/state.py` | `BattleState`（pydantic）+ `to_vector()`，维度固定 173 |
| `src/arknights_agent/perception/pipeline.py` | `Perception.read(frame) -> BattleState`：ROI 裁剪、并行推理、缓存与限频 |
| `src/arknights_agent/perception/dataset.py` | 标注数据 → YOLO 训练集（数据增强、train/val/holdout 切分，seed 固定） |
| `tools/label.py` | 本地标注工具（tkinter）：框选敌人、录入费用/CD 真值 |
| `tests/fixtures/perception/` | 200 帧 holdout 标注集 + `frames.sha256` |
| `tests/test_perception_metrics.py` | 指标门禁测试：mAP / 召回 / OCR 准确率低于阈值即失败 |

**状态向量定义（固定 173 维）：**

| 区段 | 维度 | 内容 |
| --- | --- | --- |
| 敌人 | 20 × 6 = 120 | 每槽 `x, y, w, h`（归一化到 0-1）、类别（0 普通 / 1 精英）、存活位 |
| 费用 | 1 | 当前费用 / 99，钳制到 0-1 |
| 干员 | 12 × 4 = 48 | 部署位是否占用、CD 剩余 / 冷却上限、部署费用 / 99、技能是否就绪 |
| 全局 | 4 | 关卡进度 / 总时长、击杀数 / 目标、漏怪数 / 3、剩余敌人估计 / 20 |

敌人数量不足 20 时补零；超过 20 时按检测置信度取前 20。维度变化视为破坏性变更，必须同时更新本表、`to_vector()` 与测试。

**任务：**

1. 用 `tools/label.py` 标注 ≥ 400 帧（其中 200 帧作为 holdout，永不参与训练）。
2. 训练 YOLOv8n：`yolo detect train data=config/yolo_01.yaml model=yolov8n.pt imgsz=640 epochs=100 seed=0`，导出 ONNX 供低延迟推理。
3. 实现 `detector.py` / `ocr.py` / `state.py` / `pipeline.py`，离线用 fixtures 驱动。
4. 写 `test_perception_metrics.py`：在 holdout 上计算并断言阈值。
5. 真机各 50 帧（主界面 / 战斗界面）验证状态合法性与延迟。

**验收标准（可自动验证）：**

1. 静态检查全绿；`pytest -m "not integration"` 全绿。
2. `pytest tests/test_perception_metrics.py`：敌人 mAP@50 ≥ 0.75；"战场存在敌人"召回 ≥ 0.90；费用整数值准确率 ≥ 0.98；CD 整数值准确率 ≥ 0.98。
3. `test_state_vector.py`：维度 = 173；补零与截断行为符合上表；全部值域在 [0, 1]；1000 帧回放 100% 通过 pydantic 校验。
4. 延迟：单帧 `Perception.read` p95 ≤ 300ms（batch=1，CPU 基准机；GPU 结果单独记录不算放行依据）。
5. 真机 50 帧状态向量合法率 100%，无异常帧。
6. `tests/fixtures/perception/frames.sha256` 入库且与文件一致；权重与原始帧不入库。

**停止条件：** `python scripts/verify.py --phase 2` 退出码 0，且 JSON 中 `enemy-map50 ≥ 0.75`、`ocr-cost-accuracy ≥ 0.98`、`state-dim == 173`。

---

## 阶段 3：决策模块

**目标：** 多头 DQN 在动作掩码约束下学会打 0-1 关卡，真机通关率 ≥ 80%。

**交付物（文件与职责）：**

| 文件 | 职责 |
| --- | --- |
| `src/arknights_agent/policy/env.py` | 环境接口：`reset() -> State`、`step(action) -> (State, reward, done, info)`；离线回放实现 + 真机实现共用同一接口 |
| `src/arknights_agent/policy/reward.py` | 奖励函数（下式），纯函数，单测覆盖边界 |
| `src/arknights_agent/policy/masking.py` | 动作合法性掩码：费用不足、CD 未好、格子占用/不可部署、已部署干员不可重复部署 |
| `src/arknights_agent/policy/qnet.py` | 多头 Q 网络：4 个动作头（动作类型 / 干员 / 格子 / 朝向）共享编码器，各头独立输出 |
| `src/arknights_agent/policy/train.py` | 训练入口：`python -m arknights_agent.policy.train --config config/train_01.yaml --seed 0` |
| `src/arknights_agent/policy/checkpoint.py` | checkpoint 门禁：`metrics.json` 达标才允许更新 `best.pt` 指针 |
| `src/arknights_agent/eval/bench.py` | 评测入口：`python -m arknights_agent.eval.bench --level 0-1 --episodes 20 --json runs/bench.json` |
| `tests/` | `test_masking.py`、`test_reward.py`、`test_env_replay.py`、`test_train_reproducibility.py`、`integration/test_level_01.py` |

**动作空间与奖励（固定，变更需同步本表与测试）：**

- 动作类型头（4）：`WAIT` / `DEPLOY` / `USE_SKILL` / `RETREAT`。
- 干员头（≤ 12）、格子头（8 × 12 = 96）、朝向头（4）。
- 组合动作合法当且仅当各头掩码同时为真；非法头的 Q 值置为 `-inf`，采样概率恒为 0。
- 奖励：`r = 10 * kills - 20 * leaks + 100 * win - 50 * lose - 0.01 * ticks - 0.05 * cost_wasted`，其中 `cost_wasted` 为费用达到上限后的溢出回合数。数值集中在 `reward.py` 常量段，便于阶段 4 调参。

**训练策略：** 先用阶段 2 的离线回放与录制轨迹做离线预训练（样本效率优先），真机只用于验证与少量微调；真机每局允许的最长等待为 5 分钟，超时判负。

**验收标准（可自动验证）：**

1. 静态检查全绿；`pytest -m "not integration"` 全绿。
2. `test_masking.py`：对 1000 组随机状态，非法动作 Q 值恒为 `-inf` 且 softmax 概率为 0；费用 / CD / 占位三类约束各有正反用例。
3. 离线收敛：seed=0、10k 步训练后，在 100 局离线评估集上平均回报 ≥ 随机策略基线 × 1.3，且 ≥ 规则策略基线 × 0.9。
4. 可复现：seed=0 连跑两次，训练回报曲线逐点差异 ≤ 1%。
5. 真机 `0-1` 关卡连续 20 局，通关 ≥ 16 局（通关率 ≥ 80%）；全程无人工干预；无单次状态不变超过 60 秒的卡死。
6. `checkpoint.py` 门禁：不达标时 `best.pt` 指针不变（单测注入低分 `metrics.json` 验证）。
7. `runs/bench.json` 记录基线数字，供阶段 4 对比。

**停止条件：** `python scripts/verify.py --phase 3` 退出码 0，且 JSON 中 `level-01-clear-rate ≥ 0.80`、`offline-vs-random ≥ 1.3`、`seed-reproducibility ≤ 0.01`。

---

## 阶段 4：自主进化循环

**目标：** Analyzer 从失败局中定位原因，Coder 修改策略代码，自动重跑验证，通过的保留、失败的自动回滚。

**交付物（文件与职责）：**

| 文件 | 职责 |
| --- | --- |
| `src/arknights_agent/evolution/analyzer.py` | 失败归因：把失败局归入预定义类别集合（费用不足 / 部署位置错误 / 技能时机 / CD 管理 / 漏怪 / 识别失败 / 超时） |
| `src/arknights_agent/evolution/coder.py` | 调用 Codex 生成补丁，只允许改白名单路径；产出 unified diff |
| `src/arknights_agent/evolution/sandbox.py` | 隔离与护栏：git worktree、路径白名单、单轮时间预算、进程超时 |
| `src/arknights_agent/evolution/loop.py` | 编排：采集失败样本 → 分析 → 生成补丁 → 应用 → 跑门禁 → keep / revert → 写报告 |
| `src/arknights_agent/evolution/report.py` | `runs/<id>/report.json` 结构化产出 |
| `tests/` | `test_allowlist.py`、`test_rollback.py`、`test_analyzer.py`、`test_loop_with_fake_coder.py`（`@pytest.mark.evolution`） |

**护栏（硬约束，全部可自动验证）：**

- Coder 可写路径白名单：`src/arknights_agent/policy/**`、`config/**`（仅训练与奖励参数）。
- 冻结区（写入即判失败）：`src/arknights_agent/perception/**`、`src/arknights_agent/eval/**`、`tests/**`、`scripts/verify.py`、`AGENTS.md`、`PLANS.md`。
- 每轮先记录冻结区文件 hash（`tests/fixtures/**`、`eval/**`、`scripts/verify.py`），循环结束复核；不一致则该轮作废并停机。
- 每轮必须重新跑完整门禁：上一阶段全部 check + 本阶段 check，禁止跳过。
- 连续 2 轮无提升则自动停机（防退化、防烧预算）。

**任务：**

1. 实现 `sandbox.py` 与白名单校验，先写测试再写实现（含越界写路径的拒绝用例）。
2. 实现 `analyzer.py`，用阶段 3 的失败局样本做离线标注集，分类准确率作为单测断言。
3. 实现 `coder.py`（含 prompt 模板与 diff 解析）与 `fake_coder` 测试替身。
4. 实现 `loop.py` 与 `report.py`，串起整条链路并落盘报告。
5. 真机跑 `--max-cycles 5 --budget-minutes 90`，人工只做启动与结果查看。

**验收标准（可自动验证）：**

1. 静态检查全绿；`pytest -m "not integration"` 全绿。
2. `test_allowlist.py`：20 组路径断言，白名单外写入 20/20 被拒绝，白名单内 20/20 被允许。
3. `test_rollback.py`：注入 5 个"故意变差"的假补丁 → keep 数为 0，工作树回到基线 commit（`git status --porcelain` 为空）。
4. 反作弊：循环前后冻结区 hash 完全一致；人为改一个 fixture 字符，循环必须判失败（有专门用例覆盖）。
5. `pytest -m evolution` 全绿（用 `fake_coder`，不真调 Codex）。
6. 真机 `python -m arknights_agent.evolution.loop --max-cycles 5 --budget-minutes 90` 无人值守跑完，产出 5 份 `report.json`，每份含失败样本、诊断类别、patch diff、评测分数、keep/revert 决策。
7. 进化有效性：5 轮内 ≥ 1 次 keep；循环结束时 0-1 通关率 ≥ 80% 且 ≥ 阶段 3 记录的基线；连续 2 轮无提升确实触发停机。

**停止条件：** `python scripts/verify.py --phase 4` 退出码 0，且 JSON 中 `allowlist-violations == 0`、`frozen-hash-unchanged == true`、`keeps ≥ 1`、`level-01-clear-rate ≥ 0.80`。

---

## 通用约定

**CI 顺序：** `ruff format --check` → `ruff check` → `mypy src` → `pytest -m "not integration"` → `scripts/verify.py --phase N`。真机相关 check 只在带模拟器的机器上跑，其余 check 在任意机器可复现。

**阈值改动规则：** 任何阈值（mAP、准确率、通关率、延迟）都是门禁的一部分，提高门槛需要一句理由写进 `docs/`；降低门槛视为放宽验收，必须单独提交并在提交信息里写明原因。

**基线管理：** 阶段 3 产出的 `runs/bench.json` 是阶段 4 的比较基准；重新采基线要显式说明并替换文件，不做隐式覆盖。

## 风险与开放问题

- **MAA 高层任务 vs 逐步操作**：MAA 内置战斗任务自带识别与决策，而阶段 3 需要逐步动作控制。阶段 1 必须先确认 MCP adapter 是否暴露截图与坐标点击原语；不暴露则需补最薄的一层工具，这是本计划最大的前置不确定性。
- **真机样本效率**：强化学习在真机上每步都要等待游戏动画，训练成本高。本计划用离线回放预训练 + 真机验证规避，代价是需要足够量的录制轨迹。
- **YOLOv8 标注成本**：≥ 400 帧标注是阶段 2 的主要人力开销；先做 100 帧验证标注流程，再扩量。
- **Coder 的改坏风险**：靠白名单 + 冻结区 hash + 自动回滚三层防护；阶段 4 落地后仍需人工抽查每轮 diff。
- **合规**：只使用 MAA 公开能力，涉及账号风险由使用者承担；真机循环期间保持停机条件可用。

## 后续（不在本计划范围）

多账号并发、剿灭与公招的独立策略、更高难度关卡迁移、策略蒸馏到小模型以降低推理成本。

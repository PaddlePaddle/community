# Task Proposal: PaddlePaddle__Paddle-62012

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-62012`
- SWE-Paddle 共建 issue：https://github.com/PaddlePaddle/Paddle/issues/79447
- 来源 PR：https://github.com/PaddlePaddle/Paddle/pull/62012
- PR 标题：`[SOT] rewrite resume function generation`
- PR head：`e506003f01ed7a9e36e56b17946c8f998c9a743b`
- `base_commit`：`1b68a51dbdc6b4e93a0c8e28df74e8d881272501`
- `gold_commit`：`e19e3c9435ee71ac844d78f98a34265ac7a73589`
- merged 时间：`2024-02-26T11:23:52+08:00`
- 你的身份：熟悉 Paddle SOT 的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 在 Tensor 条件分支处打断子图时，会错误读取分支中尚未创建的局部变量，导致继续执行函数生成失败并报出 `Can not get var`。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自真实合入的 SOT 缺陷修复，PR 描述和新增回归测试都明确复现 `Can not get var: y`。
- **代表性**：覆盖 Python 字节码控制流、变量读写分析、子图打断和 resume function 生成。
- **边界清楚**：生产改动和测试改动均取自相邻 Base/Gold 提交，可精确拆分。
- **非平凡性**：production diff 共 1274 行，横跨 9 个 Python 文件，包含核心执行器和指令分析逻辑的重构。
- **环境友好性**：原 PR 的 2 个测试文件可在 CPU 上运行，不依赖 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[jit, sot, bytecode, control-flow, graph-break]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_11_jumps.py`、`test/sot/test_analysis_inputs.py`
- 修复前 F2P：`TestCreateVarInIf::test_case` 触发 `Can not get var: y`；完整测试还会因缺少新的变量读写分析接口而失败。
- 修复后预期：PR 自带的两个测试文件共 15 个用例全部通过。
- P2P：`TestExecutor::test_simple`，保护原有条件跳转和循环行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：Python 3.11、兼容的 Paddle CPU 运行时，并从 checkout 加载 SOT Python 包。
- 最小测试命令：`bash tests/test.sh`
- 交叉验证入口：`bash cross_PR62012.sh`

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述用户可观察的错误与目标行为，不提供 Gold patch 的实现步骤。
- 环境风险：2024 年 SOT Python 源码需要与现有 Paddle CPU ABI 做轻量兼容；验证辅助层对 Base 和 Solution 完全相同，不改变被测逻辑。
- flaky 风险：测试使用本地 CPU Tensor，无网络、外部数据集、并发或随机依赖。
- 拆分风险：测试补丁是原 PR 两个测试文件的精确 diff；生产补丁包含 PR 的全部 9 个 Python production 文件，并逐一校验 Gold blob。

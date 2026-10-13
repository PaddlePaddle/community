# Task Proposal: PaddlePaddle__Paddle-64323

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-64323`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/64323
- PR 标题：`[SOT][dynamic shape] Support dynamic int input`
- `base_commit`：`34577c4c9a14a808fd6abec145f796b30ec93c60`
- merged 时间：`2024-05-19T12:57:37Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 会把整数函数参数视为固定常量并按具体值生成 guard，导致不同整数输入不断触发重新翻译，无法将其提升为可参与形状和算术计算的动态符号输入。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 PaddlePaddle 已合入的 SOT 动态形状工程 PR，并使用原 PR 新增测试作为行为 oracle。
- **代表性**：覆盖 Python 整数输入、Tensor reshape、混合算术、guard 生成、符号输入和翻译缓存复用。
- **边界清楚**：生产改动限定在 11 个 SOT Python 文件，测试为一个新增文件，Base 与 Gold 是相邻提交。
- **非平凡性**：production diff 为 283 行，需要贯通变量建模、运算分派、函数图输入、符号 IR 和缓存策略。
- **环境友好性**：测试可在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-shape, symbolic-int, guards, cache]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_dynamic_shape.py`
- 修复前预期：测试收集因缺少动态形状环境 guard 而失败。
- 修复后预期：整数输入 1 和 2 分别建立初始缓存，后续输入 3、4、5 复用动态翻译结果，总翻译次数保持为 2。
- P2P 候选：`test/sot/test_envs.py::TestBooleanEnvironmentVariable::test_bool_env_guard`，用于保护既有 SOT 环境变量 guard 行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：Python 3.11、兼容的 Paddle 3.0 CPU 运行时，并加载 checkout 中的纯 Python SOT 包。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述可观察的动态输入和缓存行为，不包含 Gold patch 的代码组织或实现步骤。
- 环境风险：交叉验证固定使用兼容的 Paddle 3.0 运行时，只从 checkout 加载 SOT 包，其他 `paddle.jit` 组件保持与 wheel 一致。
- flaky 风险：测试固定检查翻译次数，不依赖网络、并发时序或随机数值本身。
- 拆分风险：测试补丁是原 PR 新增测试文件的完整 diff；生产补丁只包含 11 个 production 文件，并逐个校验 Gold blob。

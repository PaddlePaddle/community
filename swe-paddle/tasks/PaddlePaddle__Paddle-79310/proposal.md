# Task Proposal: PaddlePaddle__Paddle-79310

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79310`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79310
- PR 标题：`[API Compatibility] Add paddle.nn.init.sparse_()`
- `base_commit`：`14f2f9df49bd9bd7fd94eb9cdef850c581243784`
- merged 时间：`2026-06-21`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

新增二维 Tensor 稀疏初始化 API `paddle.nn.init.sparse_()`，并统一动态图模式下函数式原地 initializer 的返回语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle PR #79310，不是合成任务。
- **代表性**：任务属于公共 API 新增和 API compatibility，可补充当前 SWE-Paddle 中以 bug fix 为主的任务类型。
- **边界清楚**：production change 集中在 `python/paddle/nn/init.py`，目标行为可通过独立测试直接观察。
- **非平凡性**：任务同时涉及新的稀疏初始化行为、输入约束、原地操作语义以及已有 initializer 的返回兼容性。
- **环境友好性**：目标逻辑位于 Python 层，可通过 AST overlay 和 controlled doubles 在 CPU 环境稳定验证，无需 GPU 或 Paddle source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[nn_init, sparse_initializer, api_compatibility, inplace_api]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79310_sparse_initializer.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，已有非动态图 initializer 行为应 pass；新增稀疏初始化能力和动态图原地返回语义相关测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，已有行为测试与新增行为测试均应 pass。
- P2P 候选：非动态图环境下已有函数式 initializer 的返回行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可运行 `pytest` 的 Python 环境，通过 AST overlay 执行 checkout 中的目标函数控制流，并使用 controlled doubles 提供依赖；无需 Paddle source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述用户可观察行为和兼容性要求，不提供 Gold patch 的具体修改方式。
- 环境风险：测试不依赖历史 Paddle package 的完整 import，也不要求 native extension 与 checkout 完全匹配。
- flaky 风险：随机相关行为通过 controlled doubles 稳定验证，不依赖随机命中、GPU 或外部数据。
- 拆分风险：新增 `sparse_` 与动态图原地 initializer 返回语义属于同一个已合入 API compatibility PR，并集中在同一 production 文件中，适合作为一个任务。

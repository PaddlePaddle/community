# Task Proposal: PaddlePaddle__Paddle-78728

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78728`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78728
- PR 标题：`support to detect the recompute context`
- `base_commit`：`52a2dfd7f3f7f402fe13161bd6d6bce5e9a727fa`
- merged 时间：`2026-04-22T07:52:37Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Fleet recompute 没有公开、线程安全的运行状态查询能力，下游逻辑无法判断当前调用是普通前向还是 recompute 执行。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 PaddlePaddle 已合入的 Fleet 工程 PR，并直接使用原 PR 新增测试作为行为 oracle。
- **代表性**：覆盖上下文状态、装饰器调用、异常恢复、线程隔离以及真实模型的前向、反向和优化步骤。
- **边界清楚**：生产改动仅涉及两个 recompute Python 文件，测试集中在一个新增文件，Base 与 Gold 为相邻提交。
- **非平凡性**：production diff 为 111 行，需要设计可重入调用接口、线程局部状态和现有 recompute 路径集成。
- **环境友好性**：测试可在 CPU 上完成，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, recompute, context, thread-local]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_recompute_context.py`
- 修复前预期：测试收集因缺少公开的 recompute 状态查询 API 而失败。
- 修复后预期：原 PR 的 14 个上下文及模型测试全部通过。
- P2P 候选：`test/legacy_test/test_recompute_with_tuple_input.py::TestPyLayer::test_tuple_input`，用于保护已有 recompute 输入与反向传播行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：Python 3.11、兼容的 Paddle 3.0 CPU 运行时，并加载 checkout 中的 Fleet recompute Python 包。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述外部可观察行为，不包含 Gold patch 的代码组织或实现步骤。
- 环境风险：2026 年 checkout 中部分纯 Python 代码依赖更新的 Paddle 内部接口；交叉验证脚本提供最小兼容层，并强制使用 CPU。
- flaky 风险：并发测试使用带超时的屏障同步；数值测试固定随机种子并使用容差比较。
- 拆分风险：测试补丁是原 PR 新增测试文件的精确 diff；生产补丁只包含两个生产文件，并逐一校验 Gold blob。

# Task Proposal: PaddlePaddle__Paddle-77150

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77150`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77150
- PR 标题：`[Eager] Add missing attribute copy when PyLayer grad node copied`
- `base_commit`：`bbc3fbcf1b93bb5fc2f6425ccbcda22816b7c8ab`
- merged 日期：`2025-12-30`
- 你的身份：contributor
- 后续联系人：TBD

## 2. 问题一句话

自定义 `PyLayer` 含部分不可导输出时，通过 `paddle.grad` 求梯度可能因 Eager autograd 节点复制后的状态不完整而导致进程异常退出。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 Paddle 主仓已合入的真实 bug-fix PR，不是合成问题。
- **外部行为明确**：同一段公开 API 代码在修复前异常退出，在修复后返回与原生计算图一致的梯度。
- **边界清楚**：目标集中在多输出 `PyLayer`、部分输出不可导以及 `paddle.grad` 的组合，不涉及模型文件、网络服务或硬件特性。
- **非平凡性**：正确修复需要理解 `paddle.grad` 与 `Tensor.backward()` 的路径差异、梯度节点复制以及多输出状态传播。
- **回归护栏明确**：目标 F2P 和已有 P2P 位于同一测试文件，可直接验证修复与兼容性。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[eager, autograd, pylayer, paddle_grad, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_pylayer_op.py`
- F2P nodeid：`test/legacy_test/test_pylayer_op.py::TestPyLayer::test_pylayer_with_partial_grad`
- P2P nodeid：`test/legacy_test/test_pylayer_op.py::TestPyLayer::test_simple_pylayer_multiple_output`
- 修复前预期：P2P 通过；F2P 失败或以 segmentation fault / access violation 等非零状态终止测试进程。
- 修复后预期：应用 `solution/code.patch` 并重新编译受影响的 Paddle 二进制后，两个 nodeid 均通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- GPU、网络服务和外部模型：不需要
- patch 类型：C++ core 修改 + Python legacy test
- 环境建议：solution 修改 C++ core，验证器需要 source build 或等价的增量编译环境；不能继续使用应用补丁前安装的旧 wheel。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述公开 API 行为和验收标准，不指出具体内部成员或修改行。
- 环境风险：历史 commit 的 source build 可能受平台工具链影响，但这些构建问题不属于任务本身。
- 崩溃风险：F2P 可能直接终止 Python 进程；verifier 应将非零退出归类为目标失败，而不是 flaky 或超时。
- flaky 风险：低。测试输入规模小，无外部依赖，修复后数值比较稳定。
- 拆分风险：目标修复集中在单一 Eager autograd 节点复制问题，适合作为一个样本。

# Task Proposal: PaddlePaddle__Paddle-78570

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78570`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78570
- PR 标题：`[API Compatibility] Support arg `closure` for `paddle.optimizer.optimizer.step``
- `base_commit`：`d8f60c6d12d57d653c97a6c9298f0c11b2db9b2a`
- merged 时间：`2026-04-17`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 `Optimizer.step()`、`Adam.step()` 和 `AdamW.step()` 增加可选 `closure` 参数，使其能够执行 closure、返回 loss，并继续完成参数更新。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 Paddle 的 API Compatibility PR，覆盖真实的训练代码迁移需求。
- **代表性**：`optimizer.step(closure)` 是常见 Optimizer API 形态，涉及可调用对象、梯度计算和参数更新之间的行为契约。
- **边界清楚**：目标集中在 `step` 的可选 `closure` 行为，并明确要求不传 `closure` 时保持既有行为。
- **非平凡性**：不仅要接受新参数，还需要保证 closure 的梯度计算环境、返回值和后续参数更新相互一致。
- **环境友好性**：可通过 checkout 中真实 `step` 控制流的 AST overlay 稳定验证，无需编译 Paddle 或使用 GPU。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[optimizer, API Compatibility, autograd]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr78570_optimizer_step_closure.py`
- 修复前预期：不传 `closure` 的既有更新路径通过；向 `Optimizer`、`Adam`、`AdamW` 的 `step` 传入 closure 时失败。
- 修复后预期：三类 `step` 均能执行 closure、返回其 loss，并使用 closure 产生的梯度完成参数更新；不传 closure 的行为保持兼容。
- P2P 候选：`test_p2p_step_without_closure_keeps_existing_update_path`
- F2P 候选：`test_f2p_optimizer_steps_accept_closure_and_return_loss`、`test_f2p_closure_runs_with_grad_enabled_and_supplies_update_grad`

测试使用 AST overlay 提取 checkout 中对应类的真实 `step` 方法，并在受控依赖环境中执行其控制流。测试断言调用结果、梯度状态及参数更新输入，不通过源码字符串匹配判断修复。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可运行 pytest 的 Python 环境；目标测试通过 AST overlay 执行 checkout 中真实 `step` 控制流，无需 Paddle source build。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述公开 API 行为和验收结果，不包含 Gold patch 的具体修改位置或代码改法。
- 环境风险：历史源码不整体 import；AST overlay 仅提取目标方法，并通过 controlled doubles 提供必要依赖，降低 wheel/source 版本差异影响。
- flaky 风险：无网络、无多进程、无随机竞态、无 GPU；closure 调用和梯度状态均为同步确定性检查。
- 拆分风险：PR 同时修改基础 `Optimizer`、`Adam`、`AdamW` 以及 AMP optimizer-like typing；production patch 作为同一 API compatibility 变更整体保留。

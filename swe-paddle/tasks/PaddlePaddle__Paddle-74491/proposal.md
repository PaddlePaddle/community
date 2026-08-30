# Task Proposal: PaddlePaddle__Paddle-74491

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-74491`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74491
* PR 标题：`[API compatibility] add new API paddle.Tensor.requires_grad`
* `base_commit`：`01666a6667e744874d7f7c379b2649d8bae67f09`
* merged 时间：`2025-08-13`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为动态图、静态图和 PIR 模式下的 Tensor 增加统一的 `requires_grad` 属性，使调用方能够通过兼容接口读取和设置 Tensor 是否参与梯度计算。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入 Paddle `develop` 分支的 API compatibility PR，不是人工构造的需求。
* **代表性**：该任务涉及常用的 Tensor 梯度控制接口，并要求动态图、静态图和 PIR 模式保持一致的属性行为。
* **边界清楚**：目标行为集中在 `requires_grad` 的读取、赋值、类型校验，以及它与现有 `stop_gradient` 状态之间的对应关系。
* **非平凡性**：任务需要在三类 Tensor 接口中提供一致的属性语义，同时保证现有梯度控制和 Tensor 元数据行为不受影响。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[tensor, autograd, api_compatibility, dynamic_graph, static_graph, pir]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr74491_requires_grad.py`
* 修复前预期：现有 Tensor 元数据和 `stop_gradient` 相关回归测试应通过；动态图、静态图和 PIR 模式下的 `requires_grad` 属性测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，三种模式下的 Tensor 均应满足 `requires_grad == (not stop_gradient)`；布尔赋值应正确更新梯度状态，非布尔赋值应抛出 `TypeError`，全部目标测试应通过。
* P2P 候选：三种 Tensor 路径中现有的 `stop_gradient` 状态和 `dim` 等元数据行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述 `requires_grad` 的公开属性行为、类型要求和兼容性约束，不透露属性的注册位置、具体函数名称或实现步骤。
* 环境风险：通过 AST overlay 隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle native runtime。
* flaky 风险：测试只验证确定性的布尔属性、异常类型和对象状态变化，不依赖随机数、GPU、并发或外部资源。
* 拆分风险：动态图、静态图和 PIR 三条路径共同实现同一个跨执行模式 Tensor API，适合作为一个完整任务。

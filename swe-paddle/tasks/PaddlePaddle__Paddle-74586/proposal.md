# Task Proposal: PaddlePaddle__Paddle-74586

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-74586`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74586
* PR 标题：`[API compatibility] add scatter_add api`
* `base_commit`：`e5c11eb4ab20851a6ab76bd0a85c8650b20b0692`
* merged 时间：`2025-08-21`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 Paddle 新增公开的 `scatter_add` API，使调用方能够根据索引在指定维度上，将源 Tensor 中的值累加到输入 Tensor 的对应位置。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入 Paddle `develop` 分支的 API compatibility PR，不是人工构造的需求。
* **代表性**：该任务涉及常用的 Tensor 索引累加操作、公共 API 导出，以及与其他深度学习框架的接口兼容。
* **边界清楚**：Base 中尚未提供 `scatter_add`，新增接口后的公共可见性和索引累加结果均可通过独立测试直接验证。
* **非平凡性**：任务不仅需要新增公共接口，还要正确处理指定维度、索引映射、重复索引累加，以及在输入 Tensor 原有值基础上的更新语义。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[api_compatibility, tensor, manipulation, scatter]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr74586_scatter_add.py`
* 修复前预期：现有 Tensor manipulation 接口的 P2P 测试应通过；`scatter_add` 的公共导出和索引累加行为测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，P2P 与全部 F2P 均应通过；`scatter_add` 应能够从两个公共命名空间访问，并正确执行索引累加。
* P2P 候选：现有 `index_add` 在动态图路径下的参数传递、底层调用和返回行为保持不变。
* F2P 设计：验证 `scatter_add` 能够在输入 Tensor 原有值的基础上进行累加，并正确处理不同维度和重复索引；同时验证该接口能够从 `paddle` 和 `paddle.tensor` 公共命名空间访问。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述 `scatter_add` 的公共接口、索引累加语义和兼容性要求，不透露 Gold patch 使用的内部函数、参数组合或具体修改位置。
* 环境风险：通过 AST overlay 隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle package 的完整导入和 native extension。
* flaky 风险：测试使用固定的 NumPy 输入、索引和确定性的 Tensor 操作替身，不依赖随机数、GPU、并发或异步时序。
* 拆分风险：`scatter_add` 的函数实现和两个公共命名空间导出共同构成一项完整的 API compatibility 能力，适合作为单个任务。

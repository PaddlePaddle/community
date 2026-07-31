# Task Proposal: PaddlePaddle__Paddle-79161

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79161`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79161
* PR 标题：`[API Compatibility] Add param alias for paddle.set_rng_state`
* `base_commit`：`8dd02b271734f7aae3669fe6dbcbea57d9cc9add`
* merged 时间：`2026-05-28`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 `paddle.set_rng_state` 的 `state_list` 参数增加等价别名 `new_state`，使使用不同参数名称的代码能够直接调用该接口。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle API Compatibility PR，不是人工构造的需求。
* **代表性**：该任务涉及公共 Python API 的参数别名、向后兼容以及冲突参数处理。
* **边界清楚**：production change 集中在 `paddle.set_rng_state` 的参数兼容行为，功能范围明确，可以通过独立测试直接验证。
* **非平凡性**：任务不仅需要支持新的参数名称，还要保持位置参数和原有 `state_list=` 调用方式不变，并正确处理两个等价参数同时出现的冲突情况。
* **环境友好性**：相关逻辑位于 Python 层，可以通过 AST overlay 在 CPU 环境中稳定验证，无需编译 Paddle 源码。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[random, api_compatibility, parameter_alias, argument_validation, python_only]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79161_set_rng_state_alias.py`
* 修复前预期：位置参数和 `state_list=` 关键字参数的现有测试应通过；`new_state=` 参数别名及冲突参数处理测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，P2P 与全部 F2P 均应通过；通过 `state_list=` 和 `new_state=` 设置相同状态时，应产生一致的行为。
* P2P 候选：通过位置参数、原有 `state_list=` 关键字参数以及显式 CPU `device` 参数设置随机数生成器状态的行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述参数别名、兼容调用和冲突输入等可观察行为，不透露具体实现方式、内部辅助机制或修改位置。
* 环境风险：AST overlay 可以隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle package 的完整导入。
* flaky 风险：测试只验证确定性的参数映射、异常类型和 CPU generator 状态设置行为，不依赖随机采样结果。
* 拆分风险：`new_state` 参数别名及其冲突处理共同构成 `set_rng_state` 的一项完整 API compatibility 改进，适合作为一个样本。

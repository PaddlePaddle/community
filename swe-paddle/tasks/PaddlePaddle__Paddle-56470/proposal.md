# Task Proposal: PaddlePaddle__Paddle-56470

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-56470`
* Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/55883
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/56470
* PR 标题：`[API Enhancement] No.6 support single int input in UpsamplingNearest2D and UpsamplingBilinear2D`
* `base_commit`：`3568a99c5f6ff0e5fd528d43bd283fde34fe078b`
* merged 时间：`2023-08-28`
* 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 `UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 的 `size` 参数增加单个整数输入支持，并将该值同时作为输出的高度和宽度。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle API 易用性增强 PR，不是人工构造的需求。
* **代表性**：该任务涉及公共 Layer API 的参数扩展、输入形式处理以及现有调用方式的向后兼容。
* **边界清楚**：production change 集中在 `UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 对 `size` 参数的处理，功能范围明确。
* **非平凡性**：虽然代码改动较小，但需要同时覆盖 nearest 和 bilinear 两个层，并保证整数、list、tuple 和 `scale_factor` 等参数形式能够保持一致且兼容的行为。
* **环境友好性**：相关逻辑位于 Python 层，可以在 CPU 环境中直接验证，无需编译 Paddle 源码或运行真实的插值算子。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[nn, layer_api, upsampling, interpolation, python_only, api_usability]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr56470_upsampling_single_int.py`
* 修复前预期：list、tuple 和 `scale_factor` 等现有参数形式的测试应通过；两个层使用单个整数 `size` 的测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，P2P 与全部 F2P 均应通过；单个整数应被两个层正确解释为相同的输出高度和宽度。
* P2P 候选：两个层使用 list 或 tuple 形式的 `size` 时，尺寸参数保持不变；使用 `scale_factor` 时，现有参数处理和调用行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述单个整数 `size` 的公开行为和兼容性要求，不透露参数在内部如何转换或具体修改位置。
* 环境风险：通过 AST overlay 隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle package 的完整导入和 native extension。
* flaky 风险：测试只验证确定性的尺寸参数处理和调用结果，不依赖随机数、GPU、网络或异步执行。
* 拆分风险：`UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 具有相同的参数增强目标，且修改集中在同一 production 文件中，适合作为一个完整样本。

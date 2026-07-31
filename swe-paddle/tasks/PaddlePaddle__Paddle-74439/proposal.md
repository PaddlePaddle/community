# Task Proposal: PaddlePaddle__Paddle-74439

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-74439`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74439
* PR 标题：`[API compatibility] add paddle.ravel`
* `base_commit`：`cb81162732f15ae02e82b07f8462e04b093c2464`
* merged 时间：`2025-08-08`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 Paddle 新增公开的 `ravel` API，使任意维度的 Tensor 都能够在保持元素顺序的前提下完整展平为一维结果。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已经合入 Paddle `develop` 分支的 API compatibility PR #74439，不是人工构造的需求。
* **代表性**：该任务涉及 Tensor 形状变换、公共 API 导出、动态图与静态图支持，以及不同维度输入的边界行为。
* **边界清楚**：production change 集中在三个 Python 文件中，功能范围明确；原 PR 测试覆盖 `ravel` 的前向结果、反向传播和不同维度输入。
* **非平凡性**：任务不仅需要新增公开接口，还要正确处理标量、一维、多维和空 Tensor，并保证现有 `flatten` 行为不受影响。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[tensor, manipulation, api_compatibility, flatten, python_api]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr74439_ravel.py`
* 修复前预期：现有 `flatten` 动态图路径的回归测试应通过；`ravel` 的公共导出、展平结果和边界输入测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，现有 `flatten` 测试应继续通过；`ravel` 对标量、一维、多维和空 Tensor 的测试，以及公共命名空间导出测试均应通过。
* P2P 候选：现有 `flatten` 在 dynamic/PIR 路径下的参数处理、底层调用和返回行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述 `ravel` 的公开接口、展平语义、边界输入和兼容性要求，不透露 Gold patch 的具体函数调用方式或修改位置。
* 环境风险：测试不导入历史 Paddle package 的完整运行环境，而是执行 source checkout 中相关的 Python 控制流程，避免对 native extension 的依赖。
* flaky 风险：测试使用固定输入并验证确定性的形状、元素顺序和对象行为，不依赖随机数、GPU、设备调度或异步执行。
* 拆分风险：公共命名空间导出、完整展平语义和边界输入支持共同构成 `ravel` API 的完整能力，适合作为单个任务。

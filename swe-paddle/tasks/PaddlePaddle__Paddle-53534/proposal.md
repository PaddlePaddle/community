# Task Proposal: PaddlePaddle__Paddle-53534

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-53534`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/53534
* PR 标题：`【BugFix】fix err of api to_tensor, which caused by numpy version update`
* `base_commit`：`f74237cd73c35b8a63d7981a190a302d0ebcd03f`
* merged 时间：`2023-05-08`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

修复 NumPy 1.24 升级后，`paddle.jit.to_static` 中的 `paddle.to_tensor` 无法处理包含 Tensor 或 Variable 的 list/tuple，并完善不支持输入类型的报错信息。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle BugFix PR，不是人工构造的问题。
* **代表性**：该任务涉及 `paddle.to_tensor`、`paddle.jit.to_static`、NumPy 版本兼容和 `dtype` 转换，是常见的 Python API 兼容问题。
* **边界清楚**：production change 只涉及 `python/paddle/tensor/creation.py`，原 PR 的测试改动位于 `test/dygraph_to_static/test_to_tensor.py`，生产代码和测试可以清楚分开。
* **非平凡性**：修复既要处理 NumPy 1.24 直接报错的情况，也要兼容旧版本 NumPy 生成 object array 的情况，同时不能破坏已有 Variable 和 `dtype` 转换逻辑。
* **区分度信号**：测试可以区分只处理 NumPy 异常的临时修复和完整修复。完整修复还应正确处理包含 Tensor/Variable 的 list/tuple、指定 `dtype` 的 Variable，以及不支持的输入类型。

## 4. 任务类型和标签

* 任务类型：`bug_fix`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[to_tensor, static_graph, numpy, list_tuple, dtype, python_api]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/legacy_test/test_to_tensor_numpy124_contract.py`
* 修复前预期：普通数值 list 和已有 Variable 的 `dtype` 处理测试应通过；NumPy 1.24 下包含 Tensor/Variable 的 list/tuple 转换测试，以及不支持输入类型的报错测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，P2P 与全部 F2P 均应通过。
* P2P 候选：普通浮点数 list 的默认 `dtype` 转换；已有 Variable 在不指定 `dtype` 或显式指定 `dtype` 时的现有行为。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only production change
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：正式 `instruction.md` 只描述 NumPy 升级后的报错、需要支持的输入和预期结果，不说明具体异常捕获、递归处理或代码修改位置。
* 环境风险：测试不依赖 Paddle wheel、GPU、C++ 编译、分布式运行环境或外部数据集。
* flaky 风险：测试通过可控的 NumPy 行为稳定复现 NumPy 1.24 的报错，不依赖测试机器实际安装的 NumPy 小版本。
* 拆分风险：包含 Tensor/Variable 的 list/tuple 转换、Variable 的 `dtype` 处理和不支持输入类型的报错都属于 `_to_tensor_static` 的同一问题，适合作为一个样本。

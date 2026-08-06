# Task Proposal: PaddlePaddle__Paddle-79310

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79310`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79310
* PR 标题：`[API Compatibility] Add paddle.nn.init.sparse_()`
* `base_commit`：`14f2f9df49bd9bd7fd94eb9cdef850c581243784`
* merged 时间：`2026-06-21`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

新增用于二维 Tensor 的 `paddle.nn.init.sparse_()`，并统一函数式原地 initializer 在动态图模式下的返回行为。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle PR #79310，不是人工构造的需求。
* **代表性**：该任务涉及公共初始化 API、新增稀疏初始化能力、原地操作语义以及与同类框架的接口兼容。
* **边界清楚**：production change 集中在 `python/paddle/nn/init.py`，包括新增 `sparse_` 以及调整现有函数式 initializer 的动态图返回行为，改动范围明确。
* **非平凡性**：任务既要实现二维 Tensor 的稀疏初始化和非法维度检查，也要保证原地修改后的返回对象符合预期，并且不能影响已有 initializer 的初始化结果和非动态图行为。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[nn_init, sparse_initializer, api_compatibility, inplace_api]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79310_sparse_initializer.py`
* 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，已有 initializer 的初始化行为和非动态图路径测试应通过；`sparse_` API、二维稀疏初始化、非法维度检查以及动态图返回输入 Tensor 的相关测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，已有行为测试与新增行为测试均应通过；`sparse_` 应正确修改并返回输入 Tensor，现有函数式原地 initializer 在动态图模式下也应返回各自接收的 Tensor。
* P2P 候选：现有 initializer 的数值初始化行为、参数处理方式以及非动态图路径的返回行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述稀疏初始化、原地返回和兼容性等用户可观察行为，不透露 Gold patch 中的随机索引生成方式、逐列处理流程或具体分支结构。
* 环境风险：测试不依赖历史 Paddle package 的完整导入，也不要求 native extension 与 source checkout 完全匹配。
* flaky 风险：测试通过固定输入和可控的随机操作替身验证稀疏位置及分布参数，不依赖真实随机结果、GPU 或外部数据。
* 拆分风险：新增 `sparse_` 和统一动态图下函数式原地 initializer 的返回行为属于同一个已合入的 API compatibility PR，且集中在同一 production 文件中，适合作为一个任务。

# 任务提案：PaddlePaddle__Paddle-79657

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79657`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79657
- PR 标题：`[Operator Mechanism] Fix paddle.compat.min/max gradient indexing`
- `base_commit`：`16037ff1effb88625041f9a1c540e8b2af3ab5c1`
- merged 时间：`2026-08-17`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

修复 `paddle.compat.min/max(input, dim=..., keepdim=False)` 在 GPU 上沿非末尾维度反向传播时，上游梯度 `stride` 与已扩维 `indices` 的 `rank` 不一致而导致的梯度索引错误。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：真实 Paddle PR，来自 `paddle.compat` API 与算子反向行为对齐中的 GPU 缺陷。
- **代表性**：它同时覆盖 `api_compatibility`、`operator_kernel`、`autograd` 和归约结果的梯度回写。
- **边界清楚**：目标只限 CUDA backward 在 `keepdim=False` 且非末尾轴时的索引修复，`keepdim=True`、末尾轴、CPU 路径和前向语义都不应改变。
- **非平凡性**：agent 需要处理降维后的形状对齐和梯度回写，不能只改 Python 包装层或表面 shape。
- **区分度信号**：完整修复必须同时通过 `min`/`max`、正轴/负轴和非均匀上游梯度场景。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cuda`
- 设备范围：`single_gpu`
- 模块标签：`[autograd, operator_kernel, api_compatibility, indexing]`

## 5. 验证思路

- 目标测试命令：在单卡 CUDA 环境下聚焦 `test/legacy_test/test_compat_minmax.py` 中的目标 compat 用例。
- 目标测试文件：`test/legacy_test/test_compat_minmax.py`
- 修复前预期：在 `base_commit` 上加入测试补丁后，新增 GPU backward 用例应失败，表现为输入梯度与 NumPy 回写结果不一致，而已有 elementwise、`keepdim=True` 或末尾轴行为继续通过。
- 修复后预期：应用真实修复后，新增用例在 `rank >= 3`、`keepdim=False`、非末尾正轴和等价负轴下都能通过，`min` 与 `max` 两条路径都与 NumPy 回写结果一致，前向结果不变。
- P2P 候选：同文件中已覆盖的 `min/max` elementwise 行为、`keepdim=True` 的 backward 行为、末尾轴行为，以及前向数值结果和 shape 断言。
- F2P 设计要点：`rank >= 3` 的 no-tie 输入；非末尾正轴和负轴；各位置不同的上游梯度；`min/max` 都对比 NumPy scatter 结果。

## 6. 环境与资源

- 资源需求：`single_gpu`
- Paddle 来源：`PaddlePaddle/Paddle` 在 `base_commit` 的 source build。
- 是否能提供 Docker：无。
- wheel 适用性：不适用，本任务涉及 C++ / CUDA 路径，不能只靠替换 Python 源码完成验证。
- patch 类型：含 C++ / 含 CUDA。
- 最小测试命令：在单卡 CUDA 环境下运行 `test/legacy_test/test_compat_minmax.py` 中的目标 compat 用例。
- 是否有 oracle 日志：无。

## 7. 风险自查

- 泄露风险：proposal 只描述可观察行为和验证约束，不暴露精确修改位置或实现细节。
- 环境风险：这是 CUDA backward 问题，验证依赖 Paddle 源码构建和可用单卡 GPU。
- flaky 风险：只要使用固定 no-tie 输入和非均匀上游梯度，结果应稳定，主要不确定性来自构建或 CUDA 环境差异。
- 拆分风险：`min`、`max`、正轴和负轴属于同一 backward indexing 缺陷族，不宜拆分。
- verifier 风险：如果测试退化成末尾轴、均匀梯度或低秩输入，就可能放过不完整修复，因此必须保留明确的 F2P/P2P 区分。

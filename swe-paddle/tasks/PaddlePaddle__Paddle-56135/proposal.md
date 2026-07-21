# Task Proposal: PaddlePaddle__Paddle-56135

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-56135`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/56135
- PR 标题：`[BugFix] fix bmm op bugs in static mode with dynamic shape`
- `base_commit`：`4f2cf7fbcaca52bb9625dc6be944f552ea1d71d5`
- merged 时间：`2023-08-16`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.bmm` 在 static graph 下无法正确接受和推导包含 unknown dimension 的 dynamic shape 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle BugFix PR，问题由 PaddleSOT 场景发现，不是合成任务。
- **代表性**：它同时覆盖 Python API validation、C++ infermeta、static graph、dynamic shape 和 output shape propagation。
- **边界清楚**：production change 只涉及 `bmm` 的 Python validation 与 `BmmInferMeta` 两个相互对应的实现位置。
- **非平凡性**：只放宽 Python validation 不足以完成修复；C++ infermeta 还必须在 unknown 与 known dimension 共存时校验兼容性并传播可确定的 output dimension。
- **区分度信号**：该任务要求同时修复 Python static graph validation 与 C++ infermeta 的 dynamic-shape contract，能够区分只处理单层逻辑的局部修复与真正跨层一致的完整修复。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[bmm, static_graph, dynamic_shape, infermeta, python_api, cpp]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_bmm_dynamic_shape_contract.py`
- 修复前预期：known-shape 和 known-incompatible-shape P2P 应 pass；Python static-graph dynamic-shape 与 C++ infermeta dynamic-shape 测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，P2P 与两个 F2P 均应 pass。
- P2P 候选：known compatible shapes 的 output shape，以及 known incompatible batch/inner dimensions 的错误校验。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python + C++ infermeta
- 环境建议：无需从头编译 Paddle；verifier 只编译 source checkout 中提取出的 `BmmInferMeta` lightweight harness，并复用本地 Python、pytest、setuptools 和 C++17 compiler
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 描述 unknown dimension 的可观察行为，不指出具体条件表达式或 helper 实现。
- 环境风险：需要可用的 C++17 compiler，但不需要 Paddle source build、CUDA 或 GPU。
- flaky 风险：测试使用固定 shapes，native harness 不包含并发、随机数或外部依赖。
- 拆分风险：两个 production files 共同定义同一个 `bmm` dynamic-shape contract，不适合拆成两个独立样本。

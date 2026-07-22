# Task Proposal: PaddlePaddle__Paddle-59909

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59909`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59909
- PR 标题：`[BugFix]Fix bug in parallel-mode select because of pure sharding`
- `base_commit`：`6279a6784b35d96b910053b55ff6f763153e454a`
- merged 时间：`2023-12-12`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 hybrid parallel topology 在 sharding 与 data parallel 同时启用时错误选择 `DATA_PARALLEL` 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle BugFix PR，不是合成任务。
- **代表性**：它覆盖 hybrid parallel topology、parallel degree composition 和 runtime mode selection。
- **边界清楚**：目标行为集中在 `get_parallel_mode` 对 data parallel、tensor parallel、pipeline parallel 和 sharding degree 的分类。
- **非平凡性**：修复需要正确处理多种 parallel strategy 的共存关系，同时保持已有 mode selection 行为不变。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, hybrid_parallel, topology, sharding, parallel_mode]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_hybrid_parallel_mode_selection.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，sharding 与 data parallel 共存的目标测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：data parallel、pure sharding、tensor parallel 和 pipeline parallel 的现有 mode-selection 用例。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：该样本可通过 AST 加载 source checkout 中的 `ParallelMode` 和 `get_parallel_mode`，不需要初始化 distributed process group
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 描述目标行为和验收标准，不直接指出具体条件顺序或修改行。
- 环境风险：该样本不依赖 Paddle runtime、GPU、NCCL 或 distributed launcher。
- flaky 风险：测试仅执行 pure Python mode selection，不涉及线程、进程或 collective communication。
- 拆分风险：该 PR 的目标集中在 hybrid parallel mode selection，适合作为一个样本。

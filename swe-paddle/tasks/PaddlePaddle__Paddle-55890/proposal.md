# Task Proposal: PaddlePaddle__Paddle-55890

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-55890`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/55890
- PR 标题：`[BugFix]Fix bug in vpp+ sharding/dp overlap`
- `base_commit`：`42ab2c34b3dd76b54b547613e12393413350c285`
- merged 时间：`2023-08-02`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 virtual pipeline parallel 与 sharding/data parallel gradient overlap 组合使用时，gradient communication 按错误步数尺度调度的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入 Paddle 分布式训练分支的真实 bug fix PR。
- **代表性**：它覆盖 virtual pipeline parallel、gradient accumulation 和 sharding/data parallel communication overlap 的组合语义。
- **边界清楚**：production change 仅位于 pipeline parallel 调度文件，目标行为可通过 communication side effect 稳定验证。
- **非平凡性**：修复需要区分 micro-batch 累积尺度、pipeline stage 数量和 model chunk 数量，错误替换任一维度都可能造成漏同步或错误 chunk 调度。
- **环境友好性**：测试通过 AST overlay 执行 checkout 中的真实控制流，并用 controlled buffer doubles 记录通信，无需 GPU、NCCL 或真实多进程。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, pipeline_parallel, virtual_pipeline, sharding, communication_overlap, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr55890_vpp_overlap_schedule.py`
- 修复前预期：旧逻辑在 accumulate steps 与 pipeline stage 数不同的场景中不会按 stage cadence 调度目标 chunk，也不会在正确的 model-chunk 边界完成非首 stage 的最终 flush。
- 修复后预期：stage cadence 和最终 flush 两个行为测试通过，完整目标测试通过。
- P2P 候选：在 accumulate steps、pipeline stage 数和 virtual/model chunk 数对齐的旧配置中，原有 chunk communication 行为保持不变。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 AST overlay 执行 checkout 中 `_overlap_comm_grads` 的真实控制流，并通过 controlled doubles 记录 `comm_grads` side effect，无需 source build。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务描述只说明错误调度的可观察行为，不给出具体变量替换或 Gold patch 修改步骤。
- 环境风险：Python-only AST overlay，不依赖历史 Paddle 模块整体 import。
- flaky 风险：测试不启动真实多进程通信，只验证确定性的 operation ordering 和 buffer side effect。
- 拆分风险：三个测试共同覆盖同一 overlap 调度契约，适合作为单一任务。

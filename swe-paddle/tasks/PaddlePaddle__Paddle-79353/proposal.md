# Task Proposal: PaddlePaddle__Paddle-79353

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79353`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79353
- PR 标题：`[Bug Fix] Fix p2p local_var bug`
- `base_commit`：`406c7afec699c23158e7ff62a0f1afb306e72afe`
- Merged date：`2026-06-23`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 pipeline P2P communication 在 first-stage no-communication path 下启用 overlap 时访问未初始化 `wait_handles` 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle distributed-training bug-fix PR。
- **边界清楚**：production change 仅涉及一个 Python 文件中 `recv_forward` 和 `send_backward` 的 first-stage branches。
- **可复现性**：目标分支不会发起实际 P2P communication，可通过直接调用 communication helper 进行验证，无需启动 multi-process distributed environment。
- **行为明确**：无 communication request 时，overlap path 应返回 `None` 作为 wait-handle result，而不是因访问未初始化的 local variable 而失败。
- **规模适中**：patch 较小，但需要保持 overlap 和 non-overlap paths 的既有 return contract。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- Execution backend：`cpu`
- Device scope：`cpu_only`
- Module tags：`[distributed, pipeline_parallel, p2p, overlap, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_p2p_overlap_boundary.py`
- 修复前预期：非 overlap 回归用例通过；两个 overlap stage 边界用例失败。
- 修复后预期：全部目标测试通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- Patch type：Python-only
- 最小测试命令：`bash tests/test.sh`
- 不需要 GPU、NCCL、multi-process distributed execution 或实际 communication backend

## 7. 风险自查

- 泄露风险：任务说明仅描述 observable behavior 和 return contract，不指定具体修改位置或实现方式。
- 环境风险：目标分支不执行实际 P2P communication，无需 GPU、NCCL 或 multi-process distributed environment。
- Flaky 风险：测试仅检查确定性的返回值和异常行为，不依赖 timing、network 或 device state。

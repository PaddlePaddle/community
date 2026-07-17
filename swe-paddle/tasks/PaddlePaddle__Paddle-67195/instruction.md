# 修复 Pipeline Parallel P2P send 前的 NaN/Inf 检查时序

## 详细描述

启用 `FLAGS_pp_check_naninf` 后，Pipeline Parallel 的 batched P2P communication 会检查 outgoing Tensor 是否包含 NaN 或 Inf。

当待发送 Tensor 包含 invalid value 时，当前行为可能先启动 send/recv operations，随后才抛出 `ValueError`。这会使 invalid data 已经进入 communication path，错误也无法在 communication side effect 发生前被阻止。

## 验收说明

- 启用 NaN/Inf checker 时，包含 NaN 或 Inf 的 outgoing Tensor 必须在任何 batched P2P operation 启动前被拒绝
- 同一 batch 中包含 receive operation 时，也不得在 invalid outgoing Tensor 被确认前启动 communication
- finite outgoing Tensor 和 receive-only Tensor 的现有行为不得退化
- 错误信息应能够标识当前 rank，且不得通过关闭 checker 或跳过 P2P logic 来规避失败

## 技术要求

- 熟悉 Python
- 了解 Paddle distributed communication 和 Pipeline Parallel
- 了解 batched P2P operation 与 communication side effects
- 了解 NaN/Inf validation 和 Paddle 单元测试开发流程

## Acceptance Criteria

- Invalid outgoing tensors raise `ValueError` before any send or receive operation is invoked.
- Valid outgoing tensors continue through the existing communication path.
- Receive tensors are not incorrectly validated as outgoing payloads.
- Existing behavior remains unchanged when the checker is disabled.

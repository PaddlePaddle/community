# 修复 persistent workers 场景下提前结束 epoch 后的 DataLoader 崩溃

## 详细描述

当 `paddle.io.DataLoader` 同时启用多个 workers 和 `persistent_workers=True` 时，用户可能只消费一个 epoch 的前几个 batches，然后通过 `break` 提前结束该 epoch。随后复用同一个 DataLoader 进入下一个 epoch。

在该工作流中，DataLoader 可能在恢复 batch 的 nested structure 时抛出 `IndexError: pop from empty list`，导致训练或 benchmark 在后续 epoch 中断。该问题通常与 background prefetch 同时发生，并可能在多次 epoch reuse 后出现。

## 验收说明

- 启用 `persistent_workers=True` 时，提前结束一个 epoch 后应能够继续复用同一个 DataLoader
- 后续 epoch 返回的 nested batch structure 和数据内容应保持正确
- 多次提前结束并重新开始 iteration 时，不应出现 `IndexError` 或其他 structure restoration 错误
- `persistent_workers=False` 以及完整消费 epoch 的现有行为不得退化

## 技术要求

- 熟悉 Python threading 和 multiprocessing
- 了解 Paddle DataLoader worker lifecycle、prefetch 和 iterator reset
- 了解 producer/consumer synchronization 与 FIFO semantics
- 了解 Paddle Python 单元测试开发流程

## Acceptance Criteria

- Reusing a persistent-worker DataLoader after an early epoch exit does not raise a structure-restoration error.
- Batch structure metadata remains correctly paired with each produced batch.
- Reset removes stale metadata without disconnecting active producers from consumers.
- Existing non-persistent and fully consumed iteration behavior remains unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing structure restoration.

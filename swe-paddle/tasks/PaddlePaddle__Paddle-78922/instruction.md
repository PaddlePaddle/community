# 修复 `flatten_state_dict` 的 Tensor reference leak

## 详细描述

FlexCheckpoint 的 `flatten_state_dict` 在处理 nested state dict 时，可能在调用结束后继续保留对输入 Tensor 的额外 reference， 在 repeated checkpoint saves 场景下，即使调用方已经释放 flattened state dict 和 key mapping，这些额外 reference 仍可能无法及时释放，导致 GPU memory 在每次保存后持续增长，并最终引发 OOM。当前释放过程可能依赖显式调用 `gc.collect()` 或等待 Python garbage collection 被动触发。
需要修复该问题，使 `flatten_state_dict` 调用期间产生的 temporary references 能够及时释放，同时保持现有 flatten 和 unflatten behavior 不变。

## 验收说明

- 调用方释放 flattened state dict 和 key mapping 后，不应残留对输入 Tensor 的额外 reference
- Tensor reference 的释放不应依赖显式调用 `gc.collect()`
- repeated 调用 `flatten_state_dict` 不应持续累积由该函数产生的 Tensor reference
- nested state dict 的 flattened keys 和 key mapping 应保持正确
- `unflatten_state_dict` 应能根据 key mapping 恢复原有 nested structure
- 输入 Tensor 的 object identity 和 data 不得改变
- 现有合法输入的 flatten 和 unflatten behavior 不得退化

## 技术要求

- 熟悉 Python reference counting 和 garbage collection
- 了解 reference cycle、closure 和 recursive function
- 了解 Paddle Tensor memory lifecycle
- 了解 FlexCheckpoint state dict utilities

## Acceptance Criteria

- Temporary references created during flattening are released promptly.
- Existing flattening and unflattening behavior remains unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, forcing broad garbage collection, or bypassing validation.
# 完善 flatten_state_dict 的对象生命周期管理

## 详细描述

完善 FlexCheckpoint 中 `flatten_state_dict` 对嵌套 state dict 的处理，使调用产生的临时对象能够在结果不再使用后及时释放，同时保持现有扁平化和还原行为不变。

## 验收说明

- 释放扁平化结果和映射后，不应继续保留输入 Tensor 的额外引用
- 嵌套 state dict 的扁平 key 和 key mapping 应保持正确
- `unflatten_state_dict` 应能按 mapping 还原原有嵌套结构
- 不得改变 Tensor 对象本身或复制其数据

## 技术要求

- 熟悉 Python 对象生命周期和垃圾回收机制
- 熟悉闭包、引用计数及递归函数
- 了解 Paddle Tensor 和 FlexCheckpoint state dict 工具

## Acceptance Criteria

- Temporary references created during flattening are released promptly.
- Existing flattening and unflattening behavior remains unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, forcing broad garbage collection, or bypassing validation.

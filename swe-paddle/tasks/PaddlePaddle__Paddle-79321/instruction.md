# 补齐 Layer.set_state_dict 的命名返回字段

## 详细描述

完善 `paddle.nn.Layer.set_state_dict` 的返回值语义，使其在保持历史二元组解包兼容性的同时，支持通过 `missing_keys` 和 `unexpected_keys` 命名字段访问不兼容键列表。

当用户将 state dict 加载到 Layer 时，返回结果应继续能够按 `missing_keys, unexpected_keys = layer.set_state_dict(...)` 的形式解包；同时也应允许调用方通过返回对象的 `missing_keys` 字段读取缺失键列表，并通过 `unexpected_keys` 字段读取多余键列表。

## 验收说明

- `set_state_dict` 应返回一个兼容 tuple 的对象，并提供 `missing_keys` 与 `unexpected_keys` 命名字段。
- 命名字段与对应的位置索引元素应指向同一批列表对象。
- 现有二元组解包用法应保持可用，不得破坏已有调用方代码。
- 修复不应改变参数加载、strict 检查、assign 行为，也不应改变缺失键和多余键列表的内容或顺序。
- 相关返回值文档和类型标注应与新的返回语义保持一致。

## 技术要求

- 熟悉 Python tuple 兼容返回对象和命名字段访问语义。
- 熟悉 Paddle `Layer` state dict 加载流程。
- 修复范围应局限于 state dict 加载 API 的返回值兼容语义。

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
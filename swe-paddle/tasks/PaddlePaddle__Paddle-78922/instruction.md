# 修复 `flatten_state_dict` 导致保存模型时显存泄漏的问题

## 详细描述

`flatten_state_dict` 中的 `_flatten` 会递归调用自身，因此产生循环引用。函数返回后，输入 state dict 中的 Tensor 仍然无法释放。

训练过程中每次保存模型都会调用 `flatten_state_dict`，未释放的 Tensor 会不断累积，导致显存持续上涨，最终 OOM。目前只能通过主动删除引用或调用 `gc.collect()` 才能释放。

需要避免 `_flatten` 产生循环引用，并保持现有的展开结果不变。

## 验收说明

- `flatten_state_dict` 返回后，不应继续持有输入 Tensor
- Tensor 的释放不应依赖手动调用 `gc.collect()`
- 多次保存模型时，不应因该函数导致显存持续增长
- 嵌套 state dict 展开后的内容和 key mapping 应保持不变

## 技术要求

- 熟悉 Python 引用计数和循环引用
- 了解递归函数
- 了解 Paddle Tensor 的显存释放方式
- 了解 FlexCheckpoint 的 state dict 处理逻辑

# 修复 `paddle.cdist` 处理 `0-size Tensor` 时的 shape 和求导问题

## 详细描述

当 `paddle.cdist` 的输入包含 `0-size` 维度时，当前实现存在两个问题。

一是四维及以上的输入无法正确计算输出 shape，会丢失部分 batch 维度。二是当输入需要计算梯度时，返回结果仍然是 `stop_gradient=True`，后续求导会报错 `Null autograd_meta`。

需要保证 `r1 == 0`、`r2 == 0` 或 `c1 == 0` 时，输出具有正确的 batch shape，并根据输入正确设置 `stop_gradient`。

## 验收说明

- `r1 == 0` 或 `r2 == 0` 时，输出 shape 应包含广播后的 batch shape，以及末尾的 `[r1, r2]`
- `c1 == 0` 时，输出应为 shape 正确的全零 Tensor
- 四维及以上输入不应丢失 batch shape
- 任意一个输入的 `stop_gradient=False` 时，输出也应为 `False`
- 两个输入的 `stop_gradient=True` 时，输出应为 `True`
- 对需要梯度的输出调用 `paddle.grad` 时，不应再出现 `Null autograd_meta` 错误
- 非 `0-size Tensor` 输入的现有行为保持不变

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 batch shape 广播规则
- 了解 Paddle 动态图自动微分和 `stop_gradient`

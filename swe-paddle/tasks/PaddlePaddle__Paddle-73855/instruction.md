# 修复 `paddle.nn.functional.dice_loss` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.dice_loss(input, label, epsilon)` 的输入 `input` 或 `label` 中存在大小为 `0` 的 dimension 时，当前实现在 Python 层会直接触发断言错误，拒绝处理任何包含 0 维度的输入。

典型表现包括：

- 抛出 AssertionError: "Any dimension of input and label cannot be equal to 0."
- 调用失败并无法继续执行

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
input = paddle.randn([0, 2]).astype(paddle.float64)
input.stop_gradient = False
label = paddle.randn([0, 1]).astype(paddle.int64)
label.stop_gradient = False
out = paddle.nn.functional.dice_loss(input, label, 1e-5)
```

上述调用中 `input` 的 shape 为 `[0, 2]`，`label` 的 shape 为 `[0, 1]`，不包含任何元素。按照 API semantics，当输入 tensor 的某个维度为 0 时，dice_loss 应正常完成计算并返回结果（值为 NaN），而不是在 Python 层直接抛出断言错误。

当前 Python 层在进入计算逻辑之前，会检查输入 shape 中是否存在 0 维度，如果存在则直接抛出异常，导致 0-size tensor 无法进入后续计算流程。

## 验收说明

- 当输入 tensor 的任意维度为 0 时，`paddle.nn.functional.dice_loss` 应正常完成，不再抛出断言错误
- 返回的结果应为 NaN（因为 0-size tensor 的计算结果在数学上未定义）
- 反向传播应正常工作，梯度 shape 应与输入 shape 一致
- 非 0-size tensor 输入下的 dice_loss 行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape、0-size Tensor 和动态图执行路径
- 了解 dice_loss 算子的计算语义

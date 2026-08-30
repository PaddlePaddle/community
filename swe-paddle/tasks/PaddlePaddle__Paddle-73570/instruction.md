# 修复 `paddle.masked_fill` 和 `paddle.diag` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.masked_fill(x, mask, value)` 或 `paddle.diag(x)` 的输入为 0-size Tensor 时，当前实现会直接进入底层算子，导致 0-size 输入无法正确处理、调用报错。

典型表现包括：

- kernel 在执行填充或对角线提取时崩溃或报错
- 0-size Tensor 输入无法通过 `masked_fill` 或 `diag` 算子
- 梯度计算时形状处理不正确

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# masked_fill 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(0, 3).astype('float32'))
mask = paddle.to_tensor(np.random.randint(0, 2, (0, 3)).astype('bool'))
value = paddle.to_tensor(np.array([1.0]).astype('float32'))
out = paddle.masked_fill(x, mask, value)
# 期望返回 shape 为 (0, 3) 的空 Tensor

# diag 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(10, 0).astype('float64'))
out = paddle.diag(x, offset=1)
# 期望返回正确 shape 的空 Tensor
```

上述调用中输入包含 0-size 维度。按照 API 语义，0-size Tensor 的操作应正常返回正确 shape 的空 Tensor。

当前实现会直接进入底层算子，导致 0-size 输入无法正确处理、调用报错。

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.masked_fill` 和 `paddle.diag` 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入一致
- 非 0-size Tensor 输入下的 masked_fill/diag 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor，或保持正确形状）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 masked_fill 和 diag 算子的输入输出语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改

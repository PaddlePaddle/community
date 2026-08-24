# 修复 `paddle.nn.functional.conv1d/conv2d/conv3d` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.conv1d/conv2d/conv3d` 的输入为 0-size Tensor 时，当前实现会调用报错。

典型表现包括：

- kernel 在执行卷积计算时崩溃或报错
- 0-size Tensor 输入无法通过 `conv1d/conv2d/conv3d` 算子
- InferMeta 计算出的输出形状不正确

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# conv1d 0-size tensor 输入
x = paddle.to_tensor(np.random.randn(0, 1, 2).astype('float32'))
filter = paddle.to_tensor(np.random.randn(1, 1, 2).astype('float32'))
out = paddle.nn.functional.conv1d(x, filter)
# 期望返回 shape 为 (0, 1, 1) 的全零 Tensor

# conv2d 0-size tensor 输入
x = paddle.to_tensor(np.random.random([0, 3, 4, 4]).astype('float32'))
filter = paddle.to_tensor(np.random.random([2, 3, 3, 3]).astype('float32'))
out = paddle.nn.functional.conv2d(x, filter)
# 期望返回 shape 为 (0, 2, 2, 2) 的全零 Tensor

# conv3d 0-size tensor 输入
x = paddle.to_tensor(np.random.random([4, 3, 0, 8, 8]).astype('float32'))
filter = paddle.to_tensor(np.random.random([5, 3, 3, 3, 3]).astype('float32'))
out = paddle.nn.functional.conv3d(x, filter, padding=1)
# 期望返回 shape 为 (4, 5, 0, 8, 8) 的全零 Tensor
```

上述调用中输入包含 0-size 维度。按照 API 语义，0-size Tensor 的卷积操作应正常返回正确 shape 的全零 Tensor。

当前实现会导致 0-size 输入无法正确处理、调用报错。

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.nn.functional.conv1d/conv2d/conv3d` 应正常完成，返回正确 shape 的全零 Tensor
- 输出 shape 应按照卷积参数正确推导
- 非 0-size Tensor 输入下的 conv1d/conv2d/conv3d 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为全零 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 conv1d/conv2d/conv3d 算子的输入输出语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 了解 InferMeta 的形状推导机制
- 需要从源码编译 Paddle 以验证修改

# 修复 `paddle.nn.functional.conv1d_transpose/conv2d_transpose/conv3d_transpose` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.conv1d_transpose/conv2d_transpose/conv3d_transpose` 的输入为 0-size Tensor 时，当前 CPU/GPU/XPU kernel 实现会直接进入后续计算逻辑，导致 kernel 内部对空数据执行计算或产生其他错误。同时，InferMeta 对 0-size 输入的输出形状计算也不正确。

典型表现包括：

- kernel 在执行转置卷积计算时崩溃或报错
- 0-size Tensor 输入无法通过 `conv1d_transpose/conv2d_transpose/conv3d_transpose` 算子
- InferMeta 计算出的输出形状不正确

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# conv1d_transpose 0-size tensor 输入
x = paddle.to_tensor(np.random.randn(0, 1, 2).astype('float32'))
filter = paddle.to_tensor(np.random.randn(1, 1, 2).astype('float32'))
out = paddle.nn.functional.conv1d_transpose(x, filter)
# 期望返回 shape 为 (0, 1, 3) 的全零 Tensor

# conv2d_transpose 0-size tensor 输入
x = paddle.to_tensor(np.random.random([0, 3, 4, 4]).astype('float32'))
filter = paddle.to_tensor(np.random.random([3, 2, 3, 3]).astype('float32'))
out = paddle.nn.functional.conv2d_transpose(x, filter)
# 期望返回 shape 为 (0, 2, 6, 6) 的全零 Tensor

# conv3d_transpose 0-size tensor 输入
x = paddle.to_tensor(np.random.random([4, 3, 0, 8, 8]).astype('float32'))
filter = paddle.to_tensor(np.random.random([3, 5, 3, 3, 3]).astype('float32'))
out = paddle.nn.functional.conv3d_transpose(x, filter)
# 期望返回 shape 为 (4, 5, 0, 10, 10) 的全零 Tensor
```

上述调用中输入包含 0-size 维度。按照 API 语义，0-size Tensor 的转置卷积操作应正常返回正确 shape 的全零 Tensor。

需要在 CPU/GPU/XPU kernel 层添加 0-size 早期返回处理，并修复 InferMeta 的形状计算逻辑：
- 在转置卷积前向 kernel 中，检查输入 `input.numel() == 0` 并使用 `phi::Full` 填充全零后直接返回
- 在转置卷积反向 kernel 中，检查输入 `input.numel() == 0` 并分配内存后直接返回
- 在 InferMeta 中，修复对 0-size 输入的输出形状计算，移除对 output_size 的错误检查

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.nn.functional.conv1d_transpose/conv2d_transpose/conv3d_transpose` 应正常完成，返回正确 shape 的全零 Tensor
- 输出的 shape 应与输入一致
- 非 0-size Tensor 输入下的 conv1d_transpose/conv2d_transpose/conv3d_transpose 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为全零 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 conv1d_transpose/conv2d_transpose/conv3d_transpose 算子的输入输出语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 了解 InferMeta 的形状推导机制
- 需要从源码编译 Paddle 以验证修改

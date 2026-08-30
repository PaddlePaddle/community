# 修复 `paddle.matmul` 在 y 为 1-D 张量且 transpose_y=True 时的精度问题

## 详细描述

当 `paddle.matmul(x, y, transpose_y=True)` 中 `y` 为 1-D 张量时，当前 CPU/XPU 梯度 kernel 实现存在精度问题。问题出在梯度计算时，对于 1-D 的 `y` 张量，`transpose_y` 标志的处理不正确，导致梯度计算结果与预期不符。

典型表现包括：

- 梯度计算结果与 numpy 参考实现不一致
- 在 `y` 为 1-D 且 `transpose_y=True` 时，梯度检查失败

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# x: (2, 100), y: (100,), transpose_y=True
x = np.random.random((2, 100)).astype('float64')
y = np.random.random((100,)).astype('float64')

x_tensor = paddle.to_tensor(x, stop_gradient=False)
y_tensor = paddle.to_tensor(y, stop_gradient=False)

out = paddle.matmul(x_tensor, y_tensor, transpose_y=True)
# out shape should be (2,)

out.sum().backward()
# x.grad and y.grad should match numpy reference
```

上述调用中 `y` 的 shape 为 `(100,)`，是一个 1-D 张量。当 `transpose_y=True` 时，按照 matmul 的语义，1-D 张量的转置应该被视为列向量，但在梯度计算中，这个标志的处理存在错误。

需要在梯度 kernel 中添加对 1-D `y` 张量的特殊处理：
- 在 `MatmulGradKernel` 中，当 `!transpose_x && transpose_y && y.dims().size() < 2` 时，将 `transpose_y` 设置为 `false`

## 验收说明

- 当 `y` 为 1-D 张量且 `transpose_y=True` 时，`paddle.matmul` 的前向和梯度计算应正确
- 梯度结果应与 numpy 参考实现一致
- 非 1-D `y` 张量输入下的 matmul 行为不得退化
- `transpose_x` 和 `transpose_y` 的其他组合也应正常工作

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 matmul 算子的数学语义和梯度计算
- 了解 1-D 张量在矩阵乘法中的特殊处理
- 了解 Paddle CPU/XPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改

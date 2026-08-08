# 修复 `paddle.linalg.multi_dot` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.multi_dot(x)` 的输入 `x` 中包含 0-size Tensor 时，当前 C++ kernel 实现会直接进入后续计算逻辑，导致 kernel 内部对空数据执行 BLAS 调用或产生其他错误。

典型表现包括：

- kernel 在执行 BLAS 矩阵乘法时崩溃或报错
- 0-size Tensor 输入无法通过 `multi_dot` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# multi_dot 0-size tensor 输入
A = np.random.random((2, 10)).astype('float64')
B = np.random.random((10, 0)).astype('float64')
C = np.random.random((0, 3)).astype('float64')

x_A = paddle.to_tensor(A)
x_B = paddle.to_tensor(B)
x_C = paddle.to_tensor(C)

out = paddle.linalg.multi_dot([x_A, x_B, x_C])
# 期望返回 shape 为 (2, 3) 的全零 Tensor
```

上述调用中 `B` 的 shape 为 `[10, 0]`，`C` 的 shape 为 `[0, 3]`，最终输出 shape 为 `[2, 3]`（非零大小）。按照 `numpy.linalg.multi_dot` 的语义，当输入中包含 0-size Tensor 时，该调用应正常完成：
- 如果输出 tensor 的 numel 为 0，直接返回空 Tensor
- 如果输出 tensor 的 numel 大于 0（如上述例子），应返回全零 Tensor

需要在 C++ kernel 层添加 0-size 早期返回处理：
- 在 `MultiDotKernel` 中，检查是否有任何输入 `x[i]->numel() == 0`，如果有：
  - 当输出 `out->numel() > 0` 时，用 `phi::Full` 填充全零
  - 直接返回，跳过后续 BLAS 计算
- 在 `MultiDotGradKernel` 中，检查是否有任何梯度 `dx[i]->numel() == 0`，如果有：
  - 对 numel > 0 的梯度用 `phi::Full` 填充全零
  - 直接返回，跳过后续梯度计算

## 验收说明

- 当输入中包含 0-size Tensor 时，`paddle.linalg.multi_dot` 应正常完成
- 输出 shape 应正确（按照矩阵乘法链式规则推导）
- 当输出 numel > 0 时，应返回全零 Tensor
- 当输出 numel == 0 时，应返回空 Tensor
- 非 0-size Tensor 输入下的 multi_dot 行为不得退化
- 梯度计算也应正常工作

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 multi_dot 算子的输入输出语义（多矩阵链式乘法）
- 了解 Paddle CPU/GPU kernel 的模板实现模式
- 需要从源码编译 Paddle 以验证修改

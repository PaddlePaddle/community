# 修复 `paddle.linalg.svdvals` 和 `paddle.linalg.eigvals` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.svdvals(x)` 或 `paddle.linalg.eigvals(x)` 的输入 `x` 为 0-size Tensor 时，当前 CPU kernel 实现会直接进入后续计算逻辑，导致 kernel 内部对空数据执行 LAPACK 调用或产生其他错误。

典型表现包括：

- kernel 在执行 LAPACK 分解时崩溃或报错
- 0-size Tensor 输入无法通过 `svdvals` 或 `eigvals` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# svdvals 0-size tensor 输入
x = paddle.to_tensor(np.random.random((1, 0)).astype('float64'))
out = paddle.linalg.svdvals(x)
# 期望返回 shape 为 (1,) 的空 Tensor（实际上 svdvals 对 (m,n) 输入返回 (min(m,n),) 的奇异值）

# eigvals 0-size tensor 输入
x = paddle.to_tensor(np.random.random((6, 0, 2, 2)).astype('float64'))
out = paddle.linalg.eigvals(x)
# 期望返回 shape 为 (6, 0, 2) 的空 Tensor
```

按照 `numpy.linalg.svd` 和 `numpy.linalg.eigvals` 的语义，0-size Tensor 的奇异值/特征值计算应正常返回正确 shape 的空 Tensor。

需要在 CPU kernel 层添加 0-size 早期返回处理：
- 在 `eigvals_kernel.cc` 中，分配输出内存后检查 `out->numel() == 0` 并直接返回
- 在 `svdvals_kernel.cc` 中，检查 `S->numel() == 0` 时分配内存并直接返回
- 在 `svdvals_grad_kernel_impl.h` 中，检查 `x_grad->numel() == 0` 时分配内存并直接返回

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.linalg.svdvals` 和 `paddle.linalg.eigvals` 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入一致（按照 SVD/eig 的语义推导）
- 非 0-size Tensor 输入下的 svdvals/eigvals 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 svdvals 和 eigvals 算子的输入输出语义
- 了解 Paddle CPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改

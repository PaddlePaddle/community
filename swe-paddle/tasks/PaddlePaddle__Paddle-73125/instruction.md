# 修复 `paddle.linalg.det` 和 `paddle.linalg.slogdet` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.det(x)` 或 `paddle.linalg.slogdet(x)` 的输入 `x` 为 0-size Tensor 时，当前 CPU/GPU kernel 实现会直接进入后续计算逻辑，导致 kernel 内部对空数据执行 LAPACK 调用或产生其他错误。

典型表现包括：

- kernel 在执行 LAPACK 分解时崩溃或报错
- 0-size Tensor 输入无法通过 `det` 或 `slogdet` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# det 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(0, 10, 10).astype('float64'))
out = paddle.linalg.det(x)
# 期望返回 shape 为 (0,) 的空 Tensor

# slogdet 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(0, 5, 5).astype('float64'))
sign, logabsdet = paddle.linalg.slogdet(x)
# 期望返回 sign shape 为 (0,)，logabsdet shape 为 (0,) 的空 Tensor
```

上述调用中 `x` 的 shape 为 `[0, 10, 10]` 或 `[0, 5, 5]`，输出 shape 为 `(0,)`。按照 `numpy.linalg.det` 和 `numpy.linalg.slogdet` 的语义，0-size Tensor 的行列式计算应正常返回正确 shape 的空 Tensor。

需要在 CPU/GPU kernel 层添加 0-size 早期返回处理：
- 在 `DeterminantKernel` 中，检查输出 `out->numel() == 0` 并直接返回
- 在 `SlogDeterminantKernel` 中，检查输入维度中是否包含 0，如果有则调整输出维度并直接返回
- 在对应的 GradKernel 中，检查梯度 `x_grad->numel() == 0` 并直接返回

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.linalg.det` 和 `paddle.linalg.slogdet` 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入一致（按照行列式的语义推导）
- 非 0-size Tensor 输入下的 det/slogdet 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 det 和 slogdet 算子的输入输出语义
- 了解 Paddle CPU/GPU kernel 的模板实现模式
- 需要从源码编译 Paddle 以验证修改

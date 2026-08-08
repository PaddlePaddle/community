# 修复 `paddle.linalg.pinv` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.pinv(x)` 的输入 `x` 为 0-size Tensor 时，当前 Python 层实现在两条代码路径中均无法正确处理：

1. **`hermitian=False` 路径（SVD）**：对 0-size Tensor 执行 SVD 后，奇异值 `s` 的最后一维为 0，此时调用 `_C_ops.max(s, [-1], True)` 会在空张量上取最大值，导致报错或产生不正确结果。
2. **`hermitian=True` 路径（eigh）**：对 0-size Tensor 执行 `eigh` 同样会失败，但实际上 hermitian 情况下 0-size 输入的伪逆就是转置末尾两个维度即可。

典型表现包括：

- 在 `hermitian=False` 时，`_C_ops.max` 对空奇异值张量报错
- 在 `hermitian=True` 时，`_C_ops.eigh` 对 0-size 输入报错

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# hermitian=False, 0-size tensor
x = paddle.to_tensor(np.random.random((0, 4, 5)).astype('float64'))
out = paddle.linalg.pinv(x, rcond=1e-15, hermitian=False)
# 期望返回 shape 为 (0, 5, 4) 的空 Tensor

# hermitian=True, 0-size tensor
x_complex = paddle.to_tensor(
    np.random.random((3, 0, 5)).astype('float32')
    + 1j * np.random.random((3, 0, 5)).astype('float32')
)
out = paddle.linalg.pinv(x_complex, rcond=1e-15, hermitian=True)
# 期望返回转置末尾两维的结果
```

按照 `numpy.linalg.pinv` 的语义，0-size Tensor 的伪逆应正常返回正确 shape 的空 Tensor。

需要在 `pinv` 函数中：
- 在 `hermitian=False` 分支中，判断 `s.shape[-1] == 0` 时对 `max_singular_val` 做特殊处理
- 在 `hermitian=True` 分支中，在动态模式下判断 `x.size == 0` 时直接返回转置末尾两维的结果

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.linalg.pinv` 应正常完成，返回与 `numpy.linalg.pinv` 一致的结果
- `hermitian=False` 和 `hermitian=True` 两条路径均需正确处理 0-size 输入
- 非 0-size Tensor 输入下的 pinv 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）
- 原有 `TestDivByZero` 中因除零抛异常的测试需要适配（不再抛异常）

## 技术要求

- 熟悉 Python 和 Paddle tensor 操作
- 了解 SVD、eigh 分解和伪逆的数学语义
- 了解 0-size Tensor 在 Paddle 中的行为
- 了解动态图/静态图模式的区别（`in_dynamic_mode()`）

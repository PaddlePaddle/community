# 修复 `paddle.linalg.triangular_solve` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.triangular_solve(x, y, ...)` 的输入 `x` 或 `y` 中存在大小为 `0` 的 dimension，即 `x.numel() == 0` 或 `y.numel() == 0` 时，当前实现会直接进入底层算子，导致报错或产生错误结果。

典型表现包括：

- 底层算子在处理 0-size tensor 时出现 shape 推断异常或计算错误
- 调用失败并抛出与 shape 相关的错误

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
x = paddle.to_tensor(np.random.random([0, 2, 2]).astype('float32'))
y = paddle.to_tensor(np.random.random([0, 2, 1]).astype('float32'))
out = paddle.linalg.triangular_solve(x, y, upper=False, left=True, unitriangular=False)
# 期望: out shape 为 [0, 2, 1]，正常返回空 tensor
```

上述调用中 `x` 的 shape 为 `[0, 2, 2]`，`y` 的 shape 为 `[0, 2, 1]`，不包含任何元素。按照 API semantics，当输入 tensor 的 numel 为 0 时，不存在需要求解的方程组，因此该调用应正常完成并返回正确 shape 的空 tensor。

此外，反向传播也需要正确处理 0-size tensor 的情况，确保梯度被正确填充。

## 验收说明

- 当输入 tensor 的 numel 为 0 时，`paddle.linalg.triangular_solve` 前向应正常完成，返回正确 shape 的空 tensor
- 返回的 out tensor 应保持与输入相同的 dtype
- 反向传播时，当 `out.numel() == 0`，`dx` 和 `dy` 应被正确填充为 0
- 非 0-size tensor 输入下的 triangular_solve 行为不得退化

## 技术要求

- 熟悉 C++ 和 Paddle phi kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 triangular_solve 算子的前向和反向语义
- 了解 Paddle phi kernel 的 CPU/GPU 实现结构

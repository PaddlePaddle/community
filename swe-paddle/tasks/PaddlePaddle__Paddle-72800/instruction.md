# 修复 `paddle.cummin` 和 `paddle.cummax` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.cummin(x, axis)` 或 `paddle.cummax(x, axis)` 的输入 `x` 为 0-size Tensor 时，导致 kernel 对空数据执行计算，产生 Segmentation fault 错误。

典型表现包括：

- kernel 在执行累积计算时崩溃或报错
- 0-size Tensor 输入无法通过 `cummin` 或 `cummax` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# cummax 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(20, 0).astype('float64'))
out, indices = paddle.cummax(x, axis=-1)
# 期望返回 out shape 为 (20, 0)，indices shape 为 (20, 0) 的空 Tensor

# cummin 0-size tensor 输入
x = paddle.to_tensor(np.random.rand(10, 0).astype('float64'))
out, indices = paddle.cummin(x, axis=-1)
# 期望返回 out shape 为 (10, 0)，indices shape 为 (10, 0) 的空 Tensor
```

上述调用中 `x` 的 shape 为 `[20, 0]` 或 `[10, 0]`，输出 shape 也为相应的 0-size shape。按照 numpy 的语义，0-size Tensor 的累积最小/最大值计算应正常返回正确 shape 的空 Tensor。

需要在 CPU/GPU kernel 层添加 0-size 早期返回处理：
- 在 cummin/cummax 前向中，检查输出并直接返回
- 在 cummin/cummax 反向中，检查梯度并直接返回

## 验收说明

- 当输入为 0-size Tensor 时，`paddle.cummin` 和 `paddle.cummax` 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入一致
- 非 0-size Tensor 输入下的 cummin/cummax 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 cummin 和 cummax 算子的输入输出语义
- 了解 Paddle CPU/GPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改

# 修复 `paddle.gather_nd` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.gather_nd(x, index)` 的 `index` 张量的最后一维为 0 时，当前 CPU/GPU/XPU kernel 实现无法正确处理这种特殊情况。根据 gather_nd 的语义，当 index 的最后一维为 0 时，输出应该是 x 的 tile 操作结果。

典型表现包括：

- kernel 在处理 0-size index 时崩溃或报错
- 0-size index 输入无法通过 `gather_nd` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# gather_nd 0-size index 输入
x = paddle.to_tensor(np.random.random([10, 20]).astype('float32'))
index = paddle.to_tensor(np.random.random([2, 0]).astype('int32'))
out = paddle.gather_nd(x, index)
# 期望返回 shape 为 (2, 10, 20) 的 Tensor
```

上述调用中 `index` 的 shape 为 `[2, 0]`，最后一维为 0。按照 gather_nd 的语义：
- 输出 shape = Index.shape[:-1] + X.shape[Index.shape[-1]:]
- 即 [2] + [10, 20] = [2, 10, 20]

需要修复 CPU/GPU/XPU kernel 对这种 0-size index 输入的支持，使 `paddle.gather_nd` 能正确处理 index 最后一维为 0 的情况，返回符合 gather_nd 语义的输出 shape 和数值。

## 验收说明

- 当 index 的最后一维为 0 时，`paddle.gather_nd` 应正常完成，返回正确 shape 的 Tensor
- 输出的 shape 应符合 gather_nd 的语义规则
- 非 0-size index 输入下的 gather_nd 行为不得退化
- 梯度计算也应正常工作

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 gather_nd 算子的输入输出语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改

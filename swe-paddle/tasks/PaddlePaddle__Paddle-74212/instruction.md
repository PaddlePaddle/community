# 修复 `paddle.multiplex` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.multiplex(inputs, index)` 的所有输入 `inputs` 均为 0-size Tensor（即 `out.numel() == 0`）时，当前 CPU/GPU kernel 实现会直接进入后续计算逻辑，导致 kernel 内部的 `PADDLE_ENFORCE_GT(ins[i]->numel(), 0, ...)` 检查失败并报错。

典型表现包括：

- kernel 抛出 `PreconditionNotMet` 错误，提示输入 numel 必须大于 0
- 0-size Tensor 输入无法通过 multiplex 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
rows = 4
index = np.array([0, 2, 2, 3]).astype('int32')
index = np.reshape(index, (rows, 1))
ins1 = np.random.random((rows, 0)).astype('float64')
ins2 = np.random.random((rows, 0)).astype('float64')
ins3 = np.random.random((rows, 0)).astype('float64')
ins4 = np.random.random((rows, 0)).astype('float64')

x1 = paddle.to_tensor(ins1)
x2 = paddle.to_tensor(ins2)
x3 = paddle.to_tensor(ins3)
x4 = paddle.to_tensor(ins4)
ids = paddle.to_tensor(index)

out = paddle.multiplex([x1, x2, x3, x4], ids)
```

上述调用中所有输入的 shape 为 `[4, 0]`，`out.numel() == 0`。按照 API semantics，当所有输入均为 0-size 时，不存在需要多路选择的数据，因此该调用应正常完成并返回正确 shape 的空 Tensor。

当前 C++ kernel 层在进入后续计算之前，没有对 0-size 输出进行显式的早期返回处理。当 `out->numel() == 0` 时，应在完成 output tensor 的 Alloc 之后，直接返回，跳过后续的 numel 检查和数据拷贝逻辑。

## 验收说明

- 当所有输入均为 0-size Tensor 时，`paddle.multiplex` kernel 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入一致（除第一维由 index 决定外）
- 非 0-size Tensor 输入下的 multiplex 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 multiplex 算子的输入输出语义
- 了解 Paddle CPU/GPU kernel 的多 backend 实现模式
- 需要从源码编译 Paddle 以验证修改

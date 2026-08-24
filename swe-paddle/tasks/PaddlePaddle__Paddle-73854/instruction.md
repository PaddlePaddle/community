# 修复 `paddle.nn.functional.instance_norm` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.instance_norm(x, ...)` 的输入 `x` 为 0-size 输入时无法正确处理、调用报错。

典型表现包括：

- `InstanceNormInferMeta` 中的 `PADDLE_ENFORCE_NE(common::product(x_dims), 0, ...)` 检查会抛出异常
- 底层 kernel 在处理 0-size tensor 时出现计算错误或崩溃
- 反向传播时 scale_grad 和 bias_grad 的维度推断错误

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
x = paddle.to_tensor(np.random.random([2, 0, 4, 5]).astype('float32'))
scale = paddle.to_tensor(np.random.random([100]).astype('float32'))
bias = paddle.to_tensor(np.random.random([100]).astype('float32'))
out = paddle.nn.functional.instance_norm(x, scale, bias)
# 期望: out shape 为 [2, 0, 4, 5]，正常返回空 tensor
```

上述调用中 `x` 的 shape 为 `[2, 0, 4, 5]`，通道数 C=0，不包含任何元素。按照 API semantics，当输入 tensor 的 numel 为 0 时，不存在需要归一化的数据，因此该调用应正常完成并返回正确 shape 的空 tensor。

当前实现会导致 0-size 输入无法正确处理、调用报错。

此外，反向传播也需要正确处理 0-size tensor 的情况。当 `x` 为 0-size 时，`d_scale` 和 `d_bias` 应被填充为 0。

## 验收说明

- 当输入 tensor 的 numel 为 0 时，`paddle.nn.functional.instance_norm` 前向应正常完成，返回正确 shape 的空 tensor
- 返回的 out tensor 应保持与输入相同的 dtype
- 反向传播时，当 `x.numel() == 0`，`d_scale` 和 `d_bias` 应被正确填充为 0
- 非 0-size tensor 输入下的 instance_norm 行为不得退化

## 技术要求

- 熟悉 C++ 和 Paddle phi kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 instance_norm 算子的前向和反向语义
- 了解 Paddle phi kernel 的 CPU/GPU/XPU 实现结构
- 了解 InferMeta 函数的维度推断机制

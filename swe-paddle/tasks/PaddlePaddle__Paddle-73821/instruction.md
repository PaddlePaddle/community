# 修复 `paddle.nn.functional.pad` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.pad(x, pad, ...)` 的输入 `x` 或 `pad` 参数为 0-size Tensor 时，当前实现无法正确处理这些情况。

典型表现包括：

- 底层 C++ kernel 在处理 0-size tensor 时崩溃或产生错误结果
- 前向 pad 操作在 0-size 输入时未使用 `pad_value` 填充输出
- 反向 pad 操作在 0-size 输入时未正确处理梯度

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# pad 0-size tensor 输入
x = paddle.to_tensor(np.random.random([0, 16]).astype('float32'))
x.stop_gradient = False
out = paddle.nn.functional.pad(
    x,
    [0, 1, 2, 3],
    mode='constant',
    value=0.5,
    pad_from_left_axis=True,
)
# 期望: out shape 为 [0, 22]，值由 pad_value 填充
out.sum().backward()
# 期望: x.grad shape 为 [0, 16]，值为 1
```

上述调用中 `x` 的 shape 为 `[0, 16]`，numel 为 0。按照 pad 的语义：
- 前向输出应保持正确的输出 shape，所有元素填充为 `pad_value`
- 反向梯度应正确传回，梯度值为 1

当前实现会导致 0-size 输入无法正确处理、调用报错。

## 验收说明

- 当输入 tensor 的 numel 为 0 时，`paddle.nn.functional.pad` 应正常完成，返回正确 shape 的 Tensor
- 前向输出的所有元素应填充为 `pad_value`
- 反向梯度应正确传回
- 当 `pad` 参数为 0-size Tensor 时，应返回 `x.clone()`
- 非 0-size tensor 输入下的 pad 行为不得退化

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 pad 算子的前向和反向语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 了解 `phi::Full` kernel 的使用
- 需要从源码编译 Paddle 以验证修改

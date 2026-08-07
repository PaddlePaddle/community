# 修复 `paddle.unique` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.unique(x)` 的输入 `x` 中存在大小为 `0` 的 dimension，即 `x.numel() == 0` 时，当前实现在动态图模式下会直接进入底层算子，导致报错。

典型表现包括：

- 底层算子在处理 0-size tensor 时 shape 推断异常
- 调用失败并抛出与 shape 相关的错误

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
x = paddle.to_tensor(np.random.randint(0, 10, (0, 2)))
out = paddle.unique(x)
```

上述调用中 `x` 的 shape 为 `[0, 2]`，不包含任何元素。按照 API semantics，当输入 tensor 的 numel 为 0 时，不存在需要去重的元素，因此该调用应正常完成并返回空 tensor。

当前 Python 层在进入底层 unique kernel 之前，没有对 0-size tensor 输入进行显式的早期返回处理。当输入 tensor 的任意维度为 0 时，应直接构造并返回正确 shape 的空 tensor，同时正确处理 `return_inverse`、`return_counts`、`return_index` 等可选返回值。

## 验收说明

- 当输入 tensor 的 numel 为 0 时，`paddle.unique` 应正常完成，返回正确 shape 的空 tensor
- 返回的 out tensor 应保持与输入相同的 dtype
- 当启用 `return_inverse`、`return_counts` 或 `return_index` 时，对应的辅助输出也应为空 tensor，dtype 为 int32 或 int64（根据 dtype 参数决定）
- 非 0-size tensor 输入下的去重行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape、0-size Tensor 和动态图执行路径
- 了解 unique 算子的多返回值语义
- 了解 Paddle 动态图和静态图执行路径的区别

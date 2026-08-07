# 修复 `static mode` 下 `paddle.bmm` 处理 `dynamic shape` 时的维度检查问题

## 详细描述

在 `static mode` 下，Tensor 的 shape 可以包含 `-1`，表示该维度需要在运行时确定。

目前，`paddle.bmm` 在检查两个输入的 batch 维度和矩阵相乘维度时，会直接比较 shape 中的数值。当一侧为 `-1`、另一侧为已知值时，本来合法的输入可能被错误地判定为维度不匹配。

例如：

```python
x = paddle.static.data(
    name="x",
    shape=[-1, 3, 4],
    dtype="float32",
)
y = paddle.static.data(
    name="y",
    shape=[2, 4, 5],
    dtype="float32",
)

out = paddle.bmm(x, y)
```

这里 `x` 的 batch 维度尚未确定，`y` 的 batch 维度为 `2`，不应在构建程序时直接报错。`out` 的 shape 应为 `[2, 3, 5]`。

矩阵相乘的两个维度也应正确处理一侧为 `-1` 的情况。只有当两侧维度都已确定且不相等时，才应报错。

## 验收说明

* 在 `static mode` 下，两个输入的 batch 维度只有一侧为 `-1` 时，`paddle.bmm` 不应报错
* 矩阵相乘的两个维度只有一侧为 `-1` 时，`paddle.bmm` 不应报错
* batch 维度一侧为 `-1`、另一侧为已知值时，输出 shape 应使用已知值
* 两侧维度都已确定且不相等时，仍应报错
* 已知且兼容的 shape 应保持原有行为

## 技术要求

* 熟悉 Python 和 C++
* 了解 Paddle `static mode`
* 了解 `dynamic shape` 和 `paddle.bmm`
* 了解算子的 shape 推导逻辑

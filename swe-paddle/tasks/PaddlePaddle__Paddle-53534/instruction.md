# 修复 NumPy 1.24 下 `paddle.jit.to_static` 中 `paddle.to_tensor` 报错的问题

## 详细描述
升级到 NumPy 1.24 后，在 `paddle.jit.to_static` 中使用 `paddle.to_tensor` 转换包含 Tensor 的 list 或 tuple 时报错。

例如：
```python
def func(x):
    a = paddle.to_tensor([1])
    b = paddle.to_tensor([2.1], dtype="int64")
    return paddle.to_tensor([a, b, [1]], dtype="float32")

out = paddle.jit.to_static(func)(x)
```

这类输入在旧版本 NumPy 中可以正常处理，但在 NumPy 1.24 中，list 转换为 NumPy array 时就直接失败了。

包含 Tensor、Variable 或嵌套 list/tuple 的输入要能够正常转换。转换结果的数值、`dtype` 和 `stop_gradient` 也要和原有行为一致。已有的 Variable 在指定不同 `dtype` 时也要正确完成类型转换。对于 dict 等不支持的输入，要返回清楚的错误信息。

## 验收说明

* NumPy 1.24 下，包含 Tensor 或 Variable 的 list/tuple 可以正常转换
* 嵌套 list/tuple 的转换结果正确
* 转换结果的数值、`dtype` 和 `stop_gradient` 符合预期
* 已有 Variable 在指定不同 `dtype` 时可以正常转换
* 普通数值、NumPy array 和标量输入的现有行为保持不变
* dict 等不支持的输入应给出明确的错误信息

## 技术要求

* 熟悉 Python 和 NumPy
* 了解 `paddle.to_tensor`
* 了解 `paddle.jit.to_static`
* 了解 Paddle 中 Tensor、Variable 和 `dtype` 的处理

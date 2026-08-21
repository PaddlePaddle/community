# 修复 `paddle.squeeze` 和 `paddle.full` 对 0-size Tensor 的支持

## 详细描述

### 1. paddle.squeeze 问题

当 `paddle.squeeze(x, axis)` 的 `axis` 参数是一个 0-size Tensor 时，当前实现会对输入无法正确处理、调用报错。

按照 API semantics 和 PyTorch 的参考行为，当 `axis` 为空 tensor 时，应该不进行任何 squeeze 操作，直接返回原始输入 `x`。

例如：

```python
import paddle

x = paddle.ones([3, 2, 1])
axis = paddle.to_tensor([], dtype=paddle.int32)  # 0-size tensor
out = paddle.squeeze(x, axis=axis)
# 期望: out.shape == [3, 2, 1]，与 x 相同
```

### 2. paddle.full 问题

当 `paddle.full(shape, fill_value)` 的 `shape` 参数是一个包含 Tensor 的列表，且其中某些 Tensor 是 0-size 的，当前实现会对输入无法正确处理、调用报错。

按照 API semantics，当 shape 列表中包含 0-size tensor 时，应该跳过该元素（相当于该维度大小为 1 或被忽略）。

例如：

```python
import paddle

out = paddle.full(
    shape=[
        paddle.to_tensor([1]),
        paddle.to_tensor([1]),
        paddle.to_tensor([]),  # 0-size tensor，应该被跳过
    ],
    fill_value=1.0,
)
# 期望: out.shape == [1, 1]
```

## 验收说明

- 当 `paddle.squeeze` 的 `axis` 为 0-size Tensor 时，应返回原始输入 `x`，不改变 shape
- 当 `paddle.full` 的 `shape` 列表中包含 0-size Tensor 时，应跳过该元素
- 输出的 shape 应与预期一致
- 非 0-size Tensor 输入下的 squeeze/full 行为不得退化
- 梯度计算也应正常工作

## 技术要求

- 熟悉 Python 和 Paddle 动态图 API 开发
- 了解 Tensor 的 size 属性和 0-size Tensor 的概念
- 了解 squeeze 和 full API 的实现逻辑
- 不需要修改 C++ kernel，仅需修改 Python 层代码

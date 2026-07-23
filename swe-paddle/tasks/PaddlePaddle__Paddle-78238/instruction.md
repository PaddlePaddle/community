# 修复 `paddle.put_along_axis` 和 `paddle.put_along_axis_` 对 0-size `indices` Tensor 的处理

## 详细描述

当 `paddle.put_along_axis(arr, indices, values, axis)` 或其 in-place 变体 `paddle.put_along_axis_` 的 `indices` 中存在大小为 `0` 的 dimension，即 `indices.numel() == 0` 时，当前实现会在实际执行 scatter update 之前报错。

典型表现包括：

- GPU：`CUDA error(9): cudaErrorInvalidConfiguration`
- CPU：`InvalidArgument: The value (4) of the non-singleton dimension does not match the corresponding value (0) in shape`

例如：

```python
paddle.put_along_axis(
    paddle.empty([2, 60], dtype="float32"),
    paddle.empty([2, 0], dtype="int64"),
    paddle.empty([2, 4], dtype="float32"),
    axis=1,
)
```

上述调用中的 `indices` shape 为 `[2, 0]`，不包含任何待更新元素。按照 API semantics，当 `indices.numel() == 0` 时，不存在需要执行的 index update，因此该调用应作为 no-op 正常完成。

当前 Python 层在进入 scatter kernel 之前，会先根据 `indices` 和 `values` 推断广播目标 shape，并调用 `paddle.broadcast_to` 扩展 `values`。当目标 shape 包含 0-size dimension，而 `values` 在对应 dimension 上的大小既不是 `0` 也不是 `1` 时，`expand` 的合法性检查会拒绝该广播，导致调用在进入底层 scatter kernel 前失败。

底层 scatter kernel 已能够识别 `indices.numel() == 0` 并跳过实际更新，因此 Python API 应避免在 empty update path 中执行不必要的 broadcast，并保持 `put_along_axis` 与 `put_along_axis_` 的既有接口语义。

## 验收说明

- 当 `indices.numel() == 0` 时，`paddle.put_along_axis` 和 `paddle.put_along_axis_` 应正常完成，不执行任何 index update
- out-of-place API 应返回与输入 `arr` 数据一致的结果，in-place API 应保持 `arr` 不变
- empty update path 不应因 `values` 与 `indices` 的 shape 无法广播而失败
- 非空 `indices` 下的 broadcasting、scatter update、`reduce` 及现有参数校验行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape、0-size Tensor 和 broadcast rules
- 了解 scatter update、in-place 与 out-of-place API semantics
- 了解 Paddle 动态图和静态图执行路径

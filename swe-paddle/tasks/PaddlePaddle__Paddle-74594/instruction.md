# 新增 `paddle.broadcast_shapes` API

## 详细描述

`paddle.broadcast_shape` 只能计算两个 shape 的广播结果。如果要同时处理三个或更多 shape 时，调用方只能重复调用该接口。

需要新增 `paddle.broadcast_shapes(*shapes)`，一次计算所有输入 shape 广播后的结果。每个 shape 可以使用 list 或 tuple 表示，返回值为 list。

未传入 shape 时返回 `[]`；只传入一个 shape 时，返回该 shape 对应的 list。传入多个 shape 时，按照现有广播规则计算结果，无法广播时抛出 `ValueError`。

## 验收说明

- 可以通过 `paddle.broadcast_shapes` 和 `paddle.tensor.broadcast_shapes` 调用该接口
- 支持传入任意数量的 list 或 tuple 形式的 shape
- 多个可广播的 shape 返回正确结果
- 未传入 shape 时返回 `[]`
- 只传入一个 shape 时返回对应的 list
- 空 shape 以及包含 `0` 或 `-1` 的 shape 按照现有广播规则处理
- 输入无法广播时抛出 `ValueError`
- 返回值为 list
- 现有 `paddle.broadcast_shape` 的行为保持不变

## 技术要求

- 熟悉 Python
- 了解 Tensor 的广播规则
- 了解 Paddle Python API 的定义和导出方式

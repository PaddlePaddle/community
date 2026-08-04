# 新增 `paddle.ravel` API

## 详细描述

Paddle 目前缺少 `ravel` 接口。需要新增 `paddle.ravel(input)`，将输入 Tensor 的所有维度展平成一维，元素顺序和数据类型保持不变。

同时还需要支持标量、普通 Tensor 和 `0-size Tensor`。

## 验收说明

- 可以通过 `paddle.ravel` 和 `paddle.tensor.ravel` 调用该接口
- 支持使用位置参数或 `input=` 传入 Tensor
- 返回结果应为一维 Tensor，元素顺序和数据类型与输入一致
- 标量输入的结果 shape 应为 `[1]`
- `0-size Tensor` 应返回 shape 为 `[0]` 的结果
- 现有 `paddle.flatten` 的行为保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle Tensor API
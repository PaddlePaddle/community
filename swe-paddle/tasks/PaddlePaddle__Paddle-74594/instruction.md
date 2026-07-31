# 新增 `paddle.broadcast_shapes` API

## 详细描述

Paddle 当前缺少用于同时计算多个 shape 广播结果的公开接口。调用方需要处理三个及以上 shape 时，无法通过单个 API 直接得到它们共同的广播形状。

请新增 `paddle.broadcast_shapes`，使其能够接收任意数量的 shape，并按照标准广播规则返回最终的结果形状。

该接口应正确处理无参数、单个 shape、空 shape 和多个 shape 等情况。当输入之间无法广播时，应抛出明确的异常。已有 `paddle.broadcast_shape` 的行为不得受到影响。

## 验收说明

* `paddle.broadcast_shapes` 应作为公开 API 提供，并能够从 `paddle` 和 `paddle.tensor` 命名空间访问
* API 应支持传入任意数量的 list 或 tuple 形式的 shape
* 多个可广播的 shape 应返回正确的共同广播形状
* 未传入 shape 时应返回空列表
* 仅传入一个 shape 时应返回对应的形状列表
* 空 shape 与其他 shape 一同传入时应按照广播规则正确处理
* 输入之间无法广播时应抛出 `ValueError`
* 返回值应为表示结果形状的 list
* 已有 `paddle.broadcast_shape` 的正常结果和异常行为不得发生变化

## 技术要求

* 熟悉 Tensor 广播规则和 shape 推导语义
* 熟悉 Paddle Python API 的定义及公共命名空间导出方式
* 了解 list、tuple 等 shape 表示形式的兼容处理
* 能够保证新增接口不影响已有的 shape 广播功能

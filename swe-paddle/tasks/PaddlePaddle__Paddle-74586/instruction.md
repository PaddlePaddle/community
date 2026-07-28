# 为 Paddle 增加 scatter_add 张量累加接口

## 详细描述

Paddle 当前缺少与常见张量库兼容的 `scatter_add` 公共接口。用户需要能够根据索引张量，在指定维度上把源张量中的值累加到输入张量对应位置，并从 `paddle` 与 `paddle.tensor` 公共命名空间访问该接口。重复索引应产生累加效果，而不是覆盖已有值。

## 验收说明

- `scatter_add` 应支持按指定维度根据索引将源值累加到输入张量，并正确处理重复索引。
- 该接口应能够通过 `paddle.scatter_add` 和 `paddle.tensor.scatter_add` 访问。
- 已有的张量 manipulation API 行为应保持兼容，不因新增接口发生回归。

## 技术要求

- 熟悉 Python API 设计与张量索引语义。
- 理解 scatter/add 类操作的维度、索引和累加行为。
- 能够维护 Paddle 公共 API 导出与现有 manipulation 模块兼容性。

# 支持 `paddle.msort` 沿第 0 维排序

## 详细描述

Paddle 当前缺少与常见深度学习框架兼容的 `paddle.msort` 公共接口。依赖该接口的代码无法直接在 Paddle 中执行。该接口应接收 Tensor，并返回一个形状和数据类型保持不变的结果，其中元素沿第 0 维按升序排列。

## 验收说明

- `paddle.msort` 应能够接收多维 Tensor，并沿第 0 维按升序返回排序结果。
- API 应支持通过 `input` 参数调用，并保持返回 Tensor 的形状和数据类型与输入一致。
- 已有 `paddle.sort` 的有效参数传递和返回行为必须保持不变。

## 技术要求

- 熟悉 Paddle Python Tensor API 及公开 API 暴露方式。
- 理解多维 Tensor 按指定 axis 排序的行为语义。
- 能维护新增 API 与既有排序能力之间的兼容性。

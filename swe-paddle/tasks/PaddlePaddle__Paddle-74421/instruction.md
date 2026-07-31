# 新增 `paddle.msort` API

## 详细描述

Paddle 当前缺少 `paddle.msort` 公共接口，依赖该接口的代码无法直接在 Paddle 中运行。请新增该 API，使其能够接收一个 Tensor，并沿第 0 维对元素进行升序排序。返回结果的形状和数据类型应与输入保持一致。

## 验收说明

* `paddle.msort` 应作为公开 API 提供
* API 应使用 `input` 作为输入参数
* 对多维 Tensor，应沿第 0 维按升序返回排序结果
* 返回 Tensor 的形状和数据类型应与输入保持一致
* API 应能够在动态图和静态图模式下正常使用
* 现有 `paddle.sort` 的行为不得发生变化

## 技术要求

* 熟悉 Paddle Python Tensor API 及公开 API 的注册方式
* 理解多维 Tensor 沿指定维度排序的行为
* 了解 Paddle 动态图和静态图模式下的 API 使用方式
* 能够维护新增 API 与现有排序功能之间的兼容性

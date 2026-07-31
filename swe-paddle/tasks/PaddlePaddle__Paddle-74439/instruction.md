# 新增 `paddle.ravel` API

## 详细描述

Paddle 当前缺少用于完整展平 Tensor 的 `ravel` 公共接口。调用方需要能够将任意维度的 Tensor 转换为一维 Tensor，同时保持原有的元素顺序和数据类型。需要新增 `paddle.ravel`，并使其能够处理标量、一维 Tensor、多维 Tensor，以及包含零长度维度的 Tensor。该接口应同时支持动态图和静态图模式，并保持正确的梯度传播行为。同时保持现有 `flatten` 接口的合法调用方式和行为不得受到影响。

## 验收说明

* `paddle.ravel` 应作为公开 API 提供，并能够从 `paddle` 和 `paddle.tensor` 命名空间访问
* API 应支持通过 `input` 参数传入 Tensor
* 返回结果应为一维 Tensor，元素顺序和数据类型应与输入保持一致
* 标量输入应得到只包含一个元素的一维结果
* 一维和多维 Tensor 应得到符合完整展平语义的结果
* 包含零长度维度的 Tensor 应得到形状正确的空一维结果
* API 应能够在动态图和静态图模式下正常使用
* 动态图模式下的反向传播行为应正确
* 现有 `flatten` 接口的行为不得发生变化

## 技术要求

* 熟悉 Paddle Python Tensor API 及公共接口导出方式
* 理解 Tensor 的形状、维数和完整展平语义
* 了解 Paddle 动态图、静态图和自动求导机制

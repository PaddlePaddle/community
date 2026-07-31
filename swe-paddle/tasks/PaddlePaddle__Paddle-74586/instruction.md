# 新增 `paddle.scatter_add` API

## 详细描述

Paddle 当前缺少用于按索引累加 Tensor 元素的 `scatter_add` 公共接口。为提升接口兼容性，调用方需要能够指定操作维度，并根据索引将源 Tensor 中的值累加到输入 Tensor 的对应位置。

请新增 `paddle.scatter_add`。当多个源元素指向同一位置时，应对这些值进行累加，而不是相互覆盖。返回结果应保留输入 Tensor 的形状和数据类型。

该接口应支持动态图和静态图模式，并能够从 `paddle` 和 `paddle.tensor` 公共命名空间访问。现有 Tensor 操作接口的合法调用方式和行为不得受到影响。

## 验收说明

* 提供公开的 `paddle.scatter_add` API，并能够通过 `paddle.tensor.scatter_add` 访问
* API 应支持通过 `input`、`dim`、`index` 和 `src` 参数调用
* 应根据 `index` 在指定维度上将 `src` 中的值累加到 `input` 的对应位置
* 多个索引指向同一位置时，对应值应正确累加
* 返回 Tensor 的形状和数据类型应与输入保持一致
* 有效的正维度和负维度参数均应得到正确处理
* API 应能够在动态图和静态图模式下正常使用
* 反向传播行为应保持正确
* 索引类型不合法、输入形状不兼容或索引越界时，应给出明确的异常
* 现有 Tensor manipulation API 的行为不得发生变化

## 技术要求

* 熟悉 Paddle Python Tensor API 及公共接口导出方式
* 理解 scatter 类操作的维度、索引和累加语义
* 了解 Paddle 动态图、静态图和自动求导机制

# 新增二维 Tensor 稀疏初始化 API

## 详细描述

Paddle 的函数式 initializer 接口需要支持二维 Tensor 的稀疏初始化能力。调用方应能够通过稀疏比例和标准差控制初始化结果，并保持函数式 initializer 的原地操作语义。

同时，动态图模式下现有函数式原地 initializer 的返回语义需要保持一致：完成初始化后应返回被修改的输入 Tensor。非动态图路径的既有行为应继续保持兼容。

## 验收说明

- 提供可用于二维 Tensor 的 `paddle.nn.init.sparse_()`，并正确处理稀疏比例、标准差以及非法维度输入。
- `sparse_` 应原地初始化并返回输入 Tensor；动态图下现有函数式原地 initializer 也应返回输入 Tensor 本身。
- 非动态图路径以及已有合法 initializer 用法的行为应保持不变。

## 技术要求

- 熟悉 Paddle initializer API 和 Tensor 原地操作语义。
- 了解 dynamic graph 与非 dynamic graph 执行模式的差异。
- 能够为随机初始化行为设计稳定、可重复的回归测试。

# 为 Tensor 增加 `requires_grad` 属性

## 详细描述

Paddle 目前通过 `stop_gradient` 控制 Tensor 是否参与梯度计算。为了提升 API 兼容性和用户体验，还需要提供语义对应的 `requires_grad` 属性。

需要为动态图 Tensor、传统静态图 Variable，以及 PIR 静态图中的 `paddle.pir.Value` 增加 `requires_grad`。其中，`paddle.pir.Value` 是 PIR 计算图中表示算子输入和输出数据的对象，可以理解为 PIR 静态图中的 Tensor-like 符号对象。

- `requires_grad=True` 时，`stop_gradient=False`
- `requires_grad=False` 时，`stop_gradient=True`

`requires_grad` 应支持读取和设置，并且只能赋布尔值。传入其他类型时应抛出 `TypeError`。

## 验收说明

- 动态图 Tensor、静态图 Variable 和 PIR Value 均支持 `requires_grad`
- `requires_grad` 的值始终与 `stop_gradient` 相反
- 设置 `requires_grad` 后，`stop_gradient` 应同步更新
- 为 `requires_grad` 赋非布尔值时应抛出 `TypeError`
- 现有 `stop_gradient` 的行为保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle 动态图、静态图和 PIR
- 了解 `stop_gradient` 与梯度计算的关系

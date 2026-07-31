# 为 Tensor 增加 `requires_grad` 属性

## 详细描述

Paddle 的 Tensor 当前缺少统一的 `requires_grad` 属性，依赖该接口判断或设置梯度计算状态的代码无法直接兼容。

请为动态图、静态图和 PIR 模式下的 Tensor 提供 `requires_grad` 属性。该属性应与现有的 `stop_gradient` 状态保持一致：当 Tensor 需要参与梯度计算时，`requires_grad` 为 `True`；停止梯度计算时，`requires_grad` 为 `False`。

该属性应支持读取和设置，并且不得影响现有梯度控制接口的行为。

## 验收说明

* 动态图、静态图和 PIR 模式下的 Tensor 均应支持读取和设置 `requires_grad`
* `requires_grad` 的值应始终与 `stop_gradient` 相反
* 将 `requires_grad` 设置为 `True` 或 `False` 后，Tensor 的梯度计算状态应立即同步更新
* 为 `requires_grad` 赋予非布尔值时，应抛出 `TypeError`
* 现有 `stop_gradient` 的读取、设置和梯度控制行为不得发生变化
* Tensor 现有的其他属性和合法调用方式不得受到影响

## 技术要求

* 熟悉 Python property 的读取和赋值语义
* 理解 Paddle 中 `stop_gradient` 与梯度计算状态的关系
* 了解 Paddle 动态图、静态图和 PIR 模式下的 Tensor 接口

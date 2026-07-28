# 为 Tensor 增加 requires_grad 兼容属性

## 详细描述

Paddle 的 Tensor 类对象在不同执行模式下缺少统一的 `requires_grad` 属性，因此依赖这一常见梯度控制接口的代码无法直接读取或修改张量是否参与梯度计算。该属性应在动态图、静态图和 PIR 模式下表现一致，并与现有的 `stop_gradient` 状态保持同步。

## 验收说明

- Tensor 类对象应能读取和设置 `requires_grad`，其布尔值始终与 `stop_gradient` 相反。
- 将 `requires_grad` 设置为布尔值后，应立即反映到梯度控制状态；传入非布尔值时应拒绝该赋值。
- 现有 Tensor 元数据和梯度控制相关的有效行为应保持兼容。

## 技术要求

- 熟悉 Python property getter/setter 语义。
- 理解 Paddle 中 `stop_gradient` 与梯度计算状态的关系。
- 能维护 dynamic graph、static graph 和 PIR Tensor 接口的一致性。

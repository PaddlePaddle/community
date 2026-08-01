# 修复 pipeline parallel 输出释放异常

## 详细描述

在 pipeline parallel 训练过程中，中间输出发送完成后，系统会清理不再需要的 Tensor 数据，以减少内存占用。

当前输出释放逻辑没有充分考虑 Tensor 的状态，可能会清理尚未初始化或已经执行过 inplace 操作的 Tensor。这里的 inplace 操作是指直接修改 Tensor 本身，并使其 inplace version 发生变化的操作。

请修复该问题，确保仅清理满足释放条件的中间输出，同时保持其他输出原有的释放行为。

## 验收说明

* 尚未初始化的 pipeline 中间输出 Tensor 不应被释放
* 已经执行过 inplace 操作的 pipeline 中间输出 Tensor 不应被释放
* 已初始化且未执行过 inplace 操作的 Tensor 应保持原有释放行为
* 单个 Tensor 以及 tuple/list 中的 Tensor 均应得到正确处理
* tuple/list 中的非 Tensor 元素不得受到影响
* 修复不得影响 pipeline parallel 原有的前向、反向和通信行为

## 技术要求

* 熟悉 Python
* 熟悉 Paddle dynamic graph 和 Tensor 生命周期
* 了解 pipeline parallel 中间输出的发送、保存和释放流程
* 了解 Tensor 初始化状态、inplace 操作和 inplace version

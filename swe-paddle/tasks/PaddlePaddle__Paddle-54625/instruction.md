# 修复 pipeline parallel 输出释放异常

## 详细描述

在 pipeline parallel 训练过程中，中间输出发送完成后，系统会回收不再需要的 Tensor 数据，以减少内存占用。当前输出释放逻辑未充分考虑 Tensor 的状态，可能会清理尚未初始化或已经发生原地修改的 Tensor，从而影响后续执行。请修复该问题，确保仅回收可以安全释放的中间输出，同时保持正常输出原有的释放行为。

## 验收说明

- 尚未初始化的 pipeline 中间输出 Tensor 不应被释放
- 已经发生原地修改的 pipeline 中间输出 Tensor 不应被释放
- 已初始化且未发生原地修改的 Tensor 应保持原有释放行为
- 单个 Tensor 以及 tuple/list 中的 Tensor 均应得到正确处理
- tuple/list 中的非 Tensor 元素不得受到影响
- 修复不得影响 pipeline parallel 原有的前向、反向和通信行为

## 技术要求

- 熟悉 Python
- 熟悉 Paddle dynamic graph 和 Tensor 生命周期
- 了解 pipeline parallel 中间输出的发送、保存和释放流程
- 了解 Tensor 初始化状态及原地修改语义

# 修复 model-parallel layers 中的内存泄漏

## 详细描述

在动态图模式下，模型并行相关操作被反复调用时可能产生额外的内存占用，并随着调用次数增加而不断累积。

该问题会影响长时间运行的模型并行任务。请修复这一内存泄漏，同时保持相关操作原有的功能和执行行为不变。

## 验收说明

- 重复调用 model-parallel `identity` 时，不应产生与调用次数持续相关的额外对象累积
- 重复调用 model-parallel `all-reduce` 时，不应产生与调用次数持续相关的额外对象累积
- `identity` 和 `all-reduce` 的前向行为不得发生变化
- `identity` 和 `all-reduce` 的反向传播及通信行为不得发生变化
- 静态图模式下的现有行为不得退化

## 技术要求

- 熟悉 Python
- 熟悉 Paddle dynamic graph 和 autograd
- 了解 model-parallel process group、`identity` 和 `all-reduce` 的执行语义

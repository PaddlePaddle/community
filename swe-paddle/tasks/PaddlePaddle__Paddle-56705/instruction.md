# 修复模型并行层重复创建 `PyLayer` 导致的内存泄漏

## 详细描述

模型并行的 `identity` 和 `all-reduce` 在动态图模式下使用 `paddle.autograd.PyLayer` 实现。

反复调用这两个操作时，会不断创建新的 `PyLayer` 类。长时间训练过程中，这些类对象会持续累积并造成内存泄漏。

需要避免重复调用带来的 `PyLayer` 类累积，同时保持 `identity` 和 `all-reduce` 原有的前向、反向和通信行为。

## 验收说明

* 重复调用模型并行 `identity` 或 `all-reduce` 时，不应持续创建新的 `PyLayer` 类
* `identity` 原有的前向结果和反向通信行为保持不变
* `all-reduce` 原有的前向结果和反向通信行为保持不变
* 现有进程组和通信参数的用法保持不变

## 技术要求

* 熟悉 Python
* 熟悉 Paddle 动态图和 `paddle.autograd.PyLayer`
* 了解模型并行中的 `identity` 和 `all-reduce`

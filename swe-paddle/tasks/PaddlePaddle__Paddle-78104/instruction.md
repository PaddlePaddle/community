# 修复 device context 对 Tensor place 的解析

## 详细描述

在动态图模式下，Tensor 的 `place` 属性可以直接传入 `paddle.device.device` 或 `paddle.cuda.device`，用于在 Tensor 所在设备上创建上下文。

在部分场景中，`tensor.place` 返回的是 `Place` 基类对象，而不是具体的设备类型对象。现有设备解析逻辑可能无法正确处理这类输入，导致设备上下文使用错误的设备。该问题在 Tensor 位于非默认 CUDA 设备时尤为明显，设备编号可能无法得到正确保留。

需要修复设备上下文对 `tensor.place` 的处理，确保上下文使用 Tensor 实际所在的设备。

## 验收说明

* CPU Tensor 的 `place` 可以正确用于设备上下文
* CUDA Tensor 的 `place` 可以正确用于设备上下文
* Tensor 位于非默认 CUDA 设备时，其设备编号应保持不变
* 退出设备上下文后，应恢复进入上下文前的设备
* 现有字符串设备参数的行为不得退化
* 现有具体 `Place` 对象的行为不得退化

## 技术要求

* 熟悉 Python
* 了解 Paddle 的 Tensor 与 `Place`
* 了解 Paddle 的设备上下文机制
* 了解 CPU 和 CUDA 设备的表示方式

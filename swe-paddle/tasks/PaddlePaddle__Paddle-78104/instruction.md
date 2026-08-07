# 修复 `paddle.cuda.device` 无法正确处理 `tensor.place` 的问题

## 详细描述

`tensor.place` 返回的是基类 `paddle.base.libpaddle.Place`，而不是具体子类（如 `CUDAPlace`）。 当前 `_convert_to_place` 通过 `type(place) is core.Place` 判断类型，导致基类 `Place` 被误识别为需要重新初始化的对象，从而每次都会被解析为默认设备 `gpu:0`。

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

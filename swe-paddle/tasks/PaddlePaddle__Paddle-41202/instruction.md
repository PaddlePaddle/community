# 为 DataLoader 增加 worker 数量自动调优能力

## 详细描述

DataLoader 的 `num_workers` 通常需要根据机器和数据处理速度反复尝试。设置得太小，数据准备可能跟不上训练；设置得太大，又会增加不必要的进程开销。现在只能由用户手动选择，换一台机器后往往还要重新调整。

所以需要为 DataLoader 增加一个可选的自动调优能力。开启后，DataLoader 使用少量样本比较不同 worker 数量的读取速度，并选出更合适的配置。普通 DataLoader 和使用 DistributedBatchSampler 的 DataLoader 都应能够正常创建和使用。

## 验收说明

* 开启自动调优后，DataLoader 能够评估不同 worker 配置并应用合适的 worker 数量。
* 调优过程只使用有限数量的样本，并支持普通 batch sampler 和 `DistributedBatchSampler`。
* 关闭自动调优或平台不支持时，DataLoader 原有的 worker 配置行为保持不变。

## 技术要求

* 熟悉 Python 和多进程数据读取。
* 熟悉 PaddlePaddle DataLoader、Dataset 和 batch sampler。
* 了解数据读取性能测试及不同平台的兼容处理。

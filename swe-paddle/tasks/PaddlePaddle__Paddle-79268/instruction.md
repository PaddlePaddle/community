# 为 `paddle.io.DistributedBatchSampler` 增加 `DistributedSampler` 别名和 `seed` 参数

## 详细描述

Paddle 目前只有 `paddle.io.DistributedBatchSampler`，缺少 `paddle.utils.data.DistributedSampler` 调用方式。

需要为 `DistributedBatchSampler` 增加 `paddle.utils.data.DistributedSampler` 别名，并支持通过 `seed` 控制开启 `shuffle` 后的采样顺序。

相同的 `seed` 和 `epoch` 应得到相同的采样顺序，不同的 `seed` 应产生不同的顺序。未传入 `seed` 时，现有调用方式和默认行为保持不变。

## 验收说明

- 可以通过 `paddle.utils.data.DistributedSampler` 使用该接口
- `DistributedSampler` 与 `paddle.io.DistributedBatchSampler` 的行为一致
- `DistributedBatchSampler` 支持可选的 `seed` 参数
- 相同的 `seed` 和 `epoch` 应产生相同的采样顺序
- 不同的 `seed` 应产生不同的采样顺序
- `shuffle=False` 时，`seed` 不应影响采样结果
- 未传入 `seed` 的现有调用方式保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle 数据采样接口
- 了解 `seed` 和 `epoch` 对 shuffle 顺序的影响

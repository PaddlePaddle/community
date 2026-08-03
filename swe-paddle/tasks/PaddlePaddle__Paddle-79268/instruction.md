# 为 `paddle.io.DistributedBatchSampler` 增加 `DistributedSampler` 别名和 `seed` 参数

## 详细描述

Paddle 当前提供了 `paddle.io.DistributedBatchSampler`，但缺少与其对应的 `paddle.utils.data.DistributedSampler` API。

请为 `paddle.io.DistributedBatchSampler` 增加 `paddle.utils.data.DistributedSampler` 别名，并支持通过 `dataset`、`num_replicas`、`rank`、`shuffle`、`seed` 和 `drop_last` 参数进行构造。

同时，为 `DistributedBatchSampler` 增加可选的 `seed` 参数。启用 shuffle 时，相同的 seed 和 epoch 应产生一致的采样顺序，不同的 seed 应能够产生不同的采样顺序。

现有未传入 seed 的调用方式不得受到影响。

## 验收说明

* `paddle.utils.data.DistributedSampler` 应作为公开 API 提供，并出现在对应的公共导出集合中
* `DistributedSampler` 应支持 `dataset`、`num_replicas`、`rank`、`shuffle`、`seed` 和 `drop_last` 参数
* `DistributedSampler` 应保持与对应分布式批采样逻辑一致的分片、打乱和末尾数据处理行为
* `DistributedBatchSampler` 应支持可选的 `seed` 参数
* 启用 shuffle 后，相同 dataset、`num_replicas`、rank、seed 和 epoch 应产生一致的采样顺序
* 在其他配置相同的情况下，不同 seed 应能够改变采样顺序
* 未显式传入 seed 时，应保持原有的默认打乱行为
* 关闭 shuffle 时，seed 不应改变原有的分片和批次结果
* `DistributedBatchSampler` 现有构造方式和合法调用行为不得发生变化

## 技术要求

* 熟悉 Paddle data loader 和 sampler 的 Python API 组织方式
* 理解分布式场景下的 rank、副本划分和批次采样语义
* 理解 seed 与 epoch 对可重复 shuffle 的影响
* 能够保证新增参数和公共接口的向后兼容性

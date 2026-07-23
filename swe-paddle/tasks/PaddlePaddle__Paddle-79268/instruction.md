# 增加 DistributedSampler 公共入口和可配置 shuffle seed

## 详细描述

Paddle 的数据加载模块已经提供 `DistributedBatchSampler`，但兼容性接口中缺少 `paddle.utils.data.DistributedSampler` 公共入口。同时，distributed batch sampler 的 shuffle 顺序只能由 epoch 决定，调用方无法提供独立的基础随机种子。

需要提供可从 `paddle.utils.data` 访问的 distributed sampler，并允许调用方通过 seed 控制 shuffle。对于相同 dataset、rank、replica、seed 和 epoch，采样顺序应保持一致；改变 seed 应能够改变 shuffle 顺序。原有未显式传入 seed 的调用方式和非 shuffle 行为必须继续有效。

## 验收说明

- `paddle.utils.data.DistributedSampler` 应作为公共 API 可访问并出现在对应导出集合中。
- distributed sampler 应接受 dataset、replica、rank、shuffle、seed 和 drop-last 参数，并保持与对应 distributed batch sampling 行为一致。
- `DistributedBatchSampler` 应接受可选 seed，并使用 seed 与 epoch 共同确定 shuffle 顺序。
- 现有不传 seed 的构造方式、固定 rank 分片和非 shuffle batch 行为应保持不变。

## 技术要求

- 熟悉 Paddle data loader 与 sampler 的 Python API 组织方式。
- 理解 distributed rank/replica 分片和 epoch-based shuffle 语义。
- 保持新增公共 API 与现有构造参数的向后兼容性。

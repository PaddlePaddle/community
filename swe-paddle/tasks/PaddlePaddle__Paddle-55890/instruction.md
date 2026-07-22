# 修复 virtual pipeline parallel 的梯度通信调度

## 详细描述

在 virtual pipeline parallel 与 sharding 或 data parallel gradient overlap 同时启用时，当 micro-batch 累积数量与 pipeline stage 数量不一致，梯度通信可能在错误的 backward step 触发，导致部分 model chunk 的梯度未按预期同步，或在同步阶段出现状态不一致。

## 验收说明

- overlap gradient communication 应按照 pipeline stage 和 model chunk 的实际调度周期触发。
- 非首 stage 应在完成全部 model chunk 的 backward 调度后正确刷新首个 chunk 的通信。
- micro-batch 累积数量与 pipeline stage 数量一致时，已有通信行为应保持不变。

## 技术要求

- 熟悉 Python 和 Paddle 分布式训练代码。
- 理解 pipeline parallel、virtual pipeline parallel 与 gradient accumulation。
- 理解 sharding/data parallel gradient communication overlap 的调度语义。

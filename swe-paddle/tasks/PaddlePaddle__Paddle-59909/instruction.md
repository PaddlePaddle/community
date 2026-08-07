# 修复 sharding 和 data parallel 同时开启时并行模式判断错误

## 详细描述

混合并行训练支持同时开启 sharding 与 data parallel。当前存在问题：在未启用 tensor parallel、pipeline parallel 的前提下，同时开启 sharding 和 data parallel 时，get_parallel_mode() 错误返回 DATA_PARALLEL。

逻辑应调整为：未开启 tensor parallel /pipeline parallel 时，只要启用 sharding，接口就返回 SHARDING_PARALLEL；仅单独开启 data parallel 时，返回 DATA_PARALLEL。

## 验收说明

* sharding 和 data parallel 同时开启时，应返回 `SHARDING_PARALLEL`
* 仅开启 data parallel 时，应返回 `DATA_PARALLEL`
* 仅开启 sharding 时，应返回 `SHARDING_PARALLEL`
* 启用 tensor parallel 时，原有的并行模式判断保持不变
* 启用 pipeline parallel 时，原有的并行模式判断保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle 混合并行训练
* 了解 data parallel、tensor parallel、pipeline parallel 和 sharding

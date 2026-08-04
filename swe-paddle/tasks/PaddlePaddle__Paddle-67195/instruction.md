# 修复 Pipeline Parallel P2P 通信没有提前拦截 NaN/Inf 的问题

## 详细描述

在 Pipeline Parallel 中开启 `FLAGS_pp_check_naninf` 后，系统会检查待发送的 Tensor 是否包含 NaN 或 Inf。

但是现有逻辑是：检查结果要等这一批 P2P 操作执行完后才会读取，即使待发送的 Tensor 中已经出现 NaN 或 Inf，发送和接收操作仍可能先被执行，之后才抛出错误。

需要在这一批 P2P 通信开始前检查所有待发送的 Tensor，只要发现 NaN 或 Inf，就应立即报错，并且这一批发送和接收操作都不应执行。

## 验收说明

- 开启 `FLAGS_pp_check_naninf` 后，待发送的 Tensor 包含 NaN 或 Inf 时应抛出 `ValueError`
- 发现 NaN 或 Inf 后，这一批 P2P 发送和接收操作都不应执行
- 同一批次同时包含发送和接收操作时，也应先完成所有待发送 Tensor 的检查
- 接收缓冲区不应被当作待发送数据检查
- 正常 Tensor 的发送行为保持不变
- 未开启 `FLAGS_pp_check_naninf` 时，现有通信行为保持不变
- 报错信息中应包含当前进程的 rank

## 技术要求

- 熟悉 Python
- 了解 Paddle Pipeline Parallel 和 P2P 通信
- 了解 Tensor 的 NaN/Inf 检查
# 修复 Pipeline Parallel P2P 通信中的 NaN/Inf 检查

## 详细描述

在 Pipeline Parallel 中，启用 `FLAGS_pp_check_naninf` 后，系统会检查待发送的 Tensor 是否包含 NaN 或 Inf。

当前检查时机存在问题：当待发送的 Tensor 包含 NaN 或 Inf 时，P2P 通信可能已经开始，之后才抛出异常。这可能导致异常数据被发送，或者使其他参与通信的进程进入不符合预期的状态。

请修复该问题，确保在启用检查的情况下，待发送 Tensor 中的 NaN 或 Inf 能够在本批次 P2P 通信开始前被检测出来。

## 验收说明

- 启用 `FLAGS_pp_check_naninf` 后，待发送 Tensor 包含 NaN 或 Inf 时，应抛出 `ValueError`
- 检测到异常数值时，本批次中的 P2P 通信不应已经开始
- 同一批次同时包含发送和接收操作时，也应在通信开始前完成待发送 Tensor 的检查
- 不包含 NaN 或 Inf 的发送操作应保持原有行为
- 仅包含接收操作时应保持原有行为
- 未启用 `FLAGS_pp_check_naninf` 时，现有通信行为不得受到影响
- 异常信息应包含当前进程的 rank

## 技术要求

- 熟悉 Python
- 了解 Paddle 分布式通信与 Pipeline Parallel
- 了解批量 P2P 通信的基本流程
- 了解 Tensor 的 NaN/Inf 检查

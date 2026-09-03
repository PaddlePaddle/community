# 修复 `paddle.iinfo` 的无符号 64 位边界值报告

## 详细描述

`paddle.iinfo(paddle.uint64)` 应当按照无符号 64 位整数类型的语义向 Python 调用方暴露正确的边界信息。当前实现中，`uint64` 的最大值在 Python 侧返回的是有符号解释的结果（例如 `-1`），而非完整的无符号 64 位最大值。这导致 API 的可观察行为与无符号整数类型的基本语义不一致。

请修复该 API，使 `paddle.iinfo(paddle.uint64).max` 返回正确的无符号 64 位最大值 `18446744073709551615`。修复范围应限于 `uint64` 边界信息的暴露问题，不应对现有 `iinfo` 或 `finfo` 的行为产生任何影响。

## 验收说明

- `paddle.iinfo(paddle.uint64).max` 必须等于 `18446744073709551615`。
- `paddle.iinfo(paddle.uint64).max` 的返回值必须是 Python `int` 类型。
- 修复后不应改变 `paddle.iinfo` / `paddle.finfo` 对其他已有 dtype 的既有行为。

## 技术要求

- 需要了解 Python 整数类型与无符号 64 位整数边界之间的兼容性特征。
- 需要了解 Paddle 的 dtype 元数据 API（`iinfo` / `finfo`）。
- 修复范围应保持最小化，仅针对 `uint64` 边界值的暴露问题进行变更。
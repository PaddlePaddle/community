# 让 TensorDataset 支持常见的多参数写法

## 详细描述

目前 `paddle.io.TensorDataset` 需要把所有 Tensor 先放进一个 list 或 tuple 再传入。很多数据加载代码会直接使用 `TensorDataset(features, labels)` 这种写法，在 Paddle 中会报参数错误。只传一个 Tensor 时问题更隐蔽：Tensor 会被当成一组 Tensor 来处理，导致数据集长度和取出的数据结构不正确。

同时，使用 `paddle.utils.data` 的代码目前也找不到 `TensorDataset`。需要补齐这些常见调用方式，且不影响 Paddle 现有的 list/tuple 写法。

## 验收说明

- 传入多个 Tensor 时，数据集长度应取各 Tensor 的第一维，按索引读取时应返回包含对应行的 tuple。
- 直接传入单个 Tensor 时，应得到只包含一项的 tuple，数据集长度仍为该 Tensor 的第一维。
- 原有 list/tuple 调用必须继续正常工作，并且 `paddle.utils.data.TensorDataset` 应可正常访问。

## 技术要求

- 熟悉 Python 位置参数、可变参数和 decorator。
- 熟悉 Paddle `Dataset` 和 Tensor 索引行为。
- 能够使用现有 DataLoader 与 API compatibility 测试验证新旧调用方式。

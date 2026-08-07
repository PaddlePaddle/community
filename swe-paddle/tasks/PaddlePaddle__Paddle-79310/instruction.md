# 新增 `paddle.nn.init.sparse_` 并补齐动态图下的返回值

## 详细描述

`paddle.nn.init` 目前缺少用于二维 Tensor 的稀疏初始化接口。

所以需要新增 `paddle.nn.init.sparse_(tensor, sparsity, std=0.01)`。该接口直接修改输入 Tensor：先按照均值为 0、标准差为 `std` 的正态分布初始化，再将每一列中的部分元素置为 0。每列置零的元素数量为 `ceil(sparsity * 行数)`。

该接口只支持二维 Tensor，其他维度的输入应抛出 `ValueError`，并返回修改后的输入 Tensor。

此外，现有函数式初始化接口在动态图下完成初始化后，也应返回传入的 Tensor，而不是返回 `None`。非动态图模式下的现有行为保持不变。

## 验收说明

- 提供 `paddle.nn.init.sparse_`
- `sparse_` 支持 `tensor`、`sparsity` 和 `std` 参数，其中 `std` 默认为 `0.01`
- 输入为二维 Tensor 时，每列应有 `ceil(sparsity * 行数)` 个元素被置为 0
- 其余元素应按照均值为 0、标准差为 `std` 的正态分布初始化
- 输入不是二维 Tensor 时应抛出 `ValueError`
- `sparse_` 应直接修改并返回输入 Tensor
- 动态图下，现有函数式初始化接口也应返回被修改的输入 Tensor
- 现有初始化结果、参数用法以及非动态图模式下的行为保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle 的初始化接口
- 了解 Tensor 的稀疏初始化
- 了解 Paddle 动态图和非动态图模式

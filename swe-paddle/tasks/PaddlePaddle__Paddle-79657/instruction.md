# 修复 `paddle.compat.min/max` CUDA 反向传播梯度索引错误

## 详细描述

`paddle.compat.min(input, dim=..., keepdim=False)` 和 `paddle.compat.max(input, dim=..., keepdim=False)` 在 CUDA 设备上沿非末尾维度执行反向传播时，上游梯度未能被正确路由到对应的输入位置。

具体表现为：当 `keepdim=False` 且 `dim` 不是末尾维度时，反向传播产生的输入梯度与预期的梯度回写结果不一致。各位置的上游梯度被错误地映射到了输入张量的错误位置。

## 验收说明

### 必须通过的行为

- `paddle.compat.min` 和 `paddle.compat.max` 在 CUDA 上沿非末尾维度反向传播时，每个非均匀上游梯度必须被路由到其对应的输入位置，梯度结果与正确的梯度回写结果一致。
- 正轴和等价负轴在非末尾维度场景下都应正确。

### 必须保持不变的行为

- 前向计算结果和输出 shape 不应改变。
- `keepdim=True` 的反向传播行为不应改变。
- 末尾维度的反向传播行为不应改变。
- CPU 设备上的行为不应改变。
- 已有的 elementwise min/max 行为不应改变。

## 技术要求

* 熟悉 C++ 和 CUDA
* 了解 Paddle 的 `paddle.compat.min/max` API
* 了解 reduction 算子的 `keepdim` 语义和反向传播梯度回写机制

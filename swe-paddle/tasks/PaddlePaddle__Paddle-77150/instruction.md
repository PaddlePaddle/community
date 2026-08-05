# 修复 `paddle.grad` 处理 `PyLayer` 部分不可导输出时的崩溃问题

## 详细描述

`PyLayer` 的拷贝构造函数缺少部分属性的拷贝。使用 `paddle.grad` 时，`GradNode` 会被拷贝，缺失的属性会导致程序出现段错误。

该问题只会在使用 `paddle.grad`，并且 `PyLayer` 的部分输出不需要梯度时出现。`Tensor.backward()` 不会拷贝 `GradNode`，因此不受影响。

## 验收说明

- 使用 `paddle.grad` 计算梯度时，不应因为 `PyLayer` 的部分输出不需要梯度而崩溃
- 拷贝后的 `GradNode` 应保留计算梯度所需的属性
- 可导输出的梯度应保持正确
- `Tensor.backward()` 的现有行为应保持不变

## 技术要求

- 熟悉 C++
- 了解 Paddle `PyLayer`
- 了解 `paddle.grad`、`Tensor.backward()` 和 `GradNode`

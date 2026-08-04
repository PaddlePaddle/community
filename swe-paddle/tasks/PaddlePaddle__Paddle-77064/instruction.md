# 增强 allclose 的 API 兼容性

## 详细描述

完善 Paddle 的 `allclose` 公开接口兼容性，同时保持其现有数值语义和返回类型不变。

需要达成的目标：

- `paddle.allclose` 继续支持位置参数及关键字参数 `x`、`y`，并为第一个和第二个输入分别支持 `input`、`other` 参数别名。
- 原参数名与别名可以混合使用，例如 `input`/`y`、`x`/`other`，结果应与普通位置参数调用一致。
- `paddle.Tensor.allclose` 支持使用 `other` 指定待比较的 Tensor。
- 新增 `paddle.compat.allclose` 兼容入口，其输入参数名为 `input` 和 `other`，并支持 `rtol`、`atol`、`equal_nan` 与 `name`。
- `paddle.allclose` 和 Tensor 方法仍返回标量布尔 Tensor；`paddle.compat.allclose` 返回 Python `bool`。
- 保留现有的容差计算、NaN 处理、支持的数据类型、动态图和静态图行为。

## 验收说明

- 所有参数别名及其混合调用形式结果一致。
- `equal_nan=True` 和 `equal_nan=False` 保持正确语义。
- 接近与不接近的输入均能产生正确结果，且两个公开入口各自满足上述返回类型契约。
- 已有 allclose 数值、输入 Tensor 数据类型和执行模式行为不得回归。
- 不允许通过删除与目标行为无关的测试、弱化相关断言或大范围绕过校验来完成任务。

## 技术要求

- 熟悉 Python 与 C++。
- 了解 Paddle 公开 API、Tensor 方法和静态图执行机制。

## Acceptance Criteria

- The observable behavior described above is implemented.
- Existing valid allclose behavior remains unchanged.
- Do not satisfy the task by deleting unrelated tests, weakening relevant assertions, or broadly bypassing validation.

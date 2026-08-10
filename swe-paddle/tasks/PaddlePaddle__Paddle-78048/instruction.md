# 完善 hsplit、dsplit 和 vsplit 的参数名兼容

## 详细描述

目前 `paddle.hsplit`、`paddle.dsplit` 和 `paddle.vsplit` 可以使用 Paddle 原有的参数名调用，但在使用一些常见的兼容参数名时会直接报参数错误。

例如使用 `input` 传入待切分的 Tensor，或者使用 `indices`、`sections` 指定切分方式时，这三个 API 目前无法正常调用。

希望补充这些参数名的兼容支持，使用户能够使用不同的参数写法完成相同的切分操作，同时不影响现有调用方式。

## 验收说明

- `paddle.hsplit`、`paddle.dsplit` 和 `paddle.vsplit` 应支持使用 `input` 代替 `x`。
- 三个 API 应支持使用 `indices` 或 `sections` 代替 `num_or_indices`，并得到与原参数名相同的切分结果。
- 原有的位置参数以及 `x`、`num_or_indices` 参数名必须继续正常工作，已有行为不能发生变化。

## 技术要求

- 熟悉 Paddle Python API 的参数定义和关键字参数调用方式。
- 理解 `hsplit`、`dsplit`、`vsplit` 的基本切分行为以及参数含义。
- 能够使用现有 API compatibility 单测验证新旧参数写法的兼容性。

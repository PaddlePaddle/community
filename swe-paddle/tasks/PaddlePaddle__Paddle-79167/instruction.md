# 为 `paddle.random.initial_seed` 增加 `paddle.initial_seed` 别名

## 详细描述

Paddle 已经提供 `paddle.random.initial_seed`，但目前不能直接通过 `paddle.initial_seed` 调用。

需要增加 `paddle.initial_seed`，并使其与 `paddle.random.initial_seed` 指向同一个函数，避免重复实现相同功能。

新增别名后，现有随机数相关 API 的行为应保持不变。

## 验收说明

* 可以通过 `paddle.initial_seed` 调用该 API
* `paddle.initial_seed` 与 `paddle.random.initial_seed` 指向同一个函数对象
* `initial_seed` 出现在 `dir(paddle)` 的结果中
* `paddle.seed`、`paddle.manual_seed` 和 `paddle.random.initial_seed` 的现有行为保持不变

## 技术要求

* 熟悉 Python 包的 API 导出方式
* 了解 Python 函数别名
* 了解 Paddle 随机数相关 API

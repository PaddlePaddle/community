# 新增 `paddle.initial_seed` 顶层别名

## 详细描述

Paddle 已经提供 `paddle.random.initial_seed`，用于获取初始随机种子，但目前无法直接通过 Paddle 顶层命名空间访问该接口。

请新增 `paddle.initial_seed`，使其作为现有 `paddle.random.initial_seed` 的顶层别名。两个名称应指向同一个函数对象，并具有完全一致的调用行为。

新增别名后，现有随机数相关接口及其行为不得受到影响。

## 验收说明

* `paddle.initial_seed` 应能够从 Paddle 顶层命名空间访问
* `paddle.initial_seed` 与 `paddle.random.initial_seed` 应指向同一个函数对象
* `paddle.initial_seed` 应能够在 Paddle 顶层公开 API 中被正常发现
* 通过两个名称调用时，应获得一致的结果
* 现有 `paddle.seed`、`paddle.manual_seed` 和 `paddle.random.initial_seed` 的行为不得发生变化

## 技术要求

* 熟悉 Python package 的顶层 API 导出方式
* 了解 API 别名和 Python 对象身份的语义
* 了解顶层公共 API 的可访问性与可发现性
* 能够保证新增别名不影响已有随机数接口

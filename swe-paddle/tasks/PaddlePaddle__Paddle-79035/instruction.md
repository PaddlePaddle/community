# 为 `paddle.optim` 补充 `lr_scheduler` API

## 详细描述

目前 Paddle 的学习率调度器主要通过 `paddle.optimizer.lr` 提供。

在将其他框架的训练代码迁移到 Paddle 时，一些代码会使用 `paddle.optim.lr_scheduler` 下的 `StepLR`、`MultiStepLR`、`LambdaLR` 等 API。Paddle 当前缺少这些兼容 API，导致相关代码不能直接运行，需要用户手动修改导入路径和部分 API 名称。

需要在 `paddle.optim.lr_scheduler` 下补充对应的学习率调度器 API，使常见的训练代码能够更方便地迁移到 Paddle。

这些新增 API 应复用 Paddle 已有的学习率调度器实现，并保持原有 `paddle.optimizer.lr` API 的行为不变。

## 验收说明

* `paddle.optim.lr_scheduler` 应提供 `LambdaLR`、`MultiplicativeLR`、`StepLR`、`MultiStepLR`、`ConstantLR`、`LinearLR`、`ExponentialLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`、`CyclicLR`、`CosineAnnealingWarmRestarts`、`OneCycleLR` 和 `LRScheduler`。
* 新增的兼容 API 要对应 Paddle 已有的学习率调度器能力，不能重新实现一套不同的调度逻辑。
* 原有 `paddle.optimizer.lr` API 以及已有学习率调度器行为应保持不变。

## 技术要求

* 熟悉 Python 模块导入机制
* 熟悉 Paddle optimizer 和学习率调度器相关 API

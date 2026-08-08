# 为 `paddle.optim` 补充 `lr_scheduler` API

## 详细描述

目前 Paddle 的学习率调度器主要通过 `paddle.optimizer.lr` 提供，例如 `StepDecay`、`MultiStepDecay`、`LambdaDecay` 和 `ReduceOnPlateau` 等。

在迁移其他框架的训练代码到 Paddle 时，学习率调度器的导入方式和 API 名称存在差异。例如，一些代码会使用 `lr_scheduler.StepLR`、`lr_scheduler.MultiStepLR`、`lr_scheduler.LambdaLR` 等 API，而 Paddle 中对应的学习率调度器位于 `paddle.optimizer.lr` 下，部分名称也有所不同，因此迁移时需要额外修改相关代码。

为了减少迁移成本，需要为 `paddle.optim` 补充 `lr_scheduler` API，并为 Paddle 已有的学习率调度器提供对应的兼容 API 名称。新增的 API 要复用 Paddle 现有的学习率调度器实现，不改变原有计算逻辑，同时保证 `paddle.optimizer.lr` 下已有的其他 API 的使用方式和行为保持不变。

## 验收说明

* 支持通过 `paddle.optim.lr_scheduler` 使用 `LambdaLR`、`MultiplicativeLR`、`StepLR`、`MultiStepLR`、`ConstantLR`、`LinearLR`、`ExponentialLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`、`CyclicLR`、`CosineAnnealingWarmRestarts`、`OneCycleLR` 和 `LRScheduler`。
* `paddle.optim.lr_scheduler` 下新增的 API 应复用 `paddle.optimizer.lr` 中已有的学习率调度器功能。
* 原有 `paddle.optimizer.lr` API 及已有学习率调度器行为保持不变。

## 技术要求

* 熟悉 Python 模块导入机制
* 熟悉 Paddle optimizer 和学习率调度器相关 API

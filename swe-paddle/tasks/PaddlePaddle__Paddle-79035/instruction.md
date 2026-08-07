# 为 `paddle.optim` 提供 `lr_scheduler` API compatibility namespace

## 详细描述

当前 `paddle.optimizer.lr` 已提供学习率 scheduler，但用户无法通过 `paddle.optim.lr_scheduler` 使用对应的兼容命名。这会使采用另一套常见 scheduler 命名方式的代码在迁移到 Paddle 时因 namespace 或符号缺失而失败。

任务需要让 `paddle.optim.lr_scheduler` 成为可用的公开兼容入口，并提供与现有 Paddle scheduler 对应的兼容名称；同时不得改变原有 `paddle.optimizer.lr` 的有效行为。

## 验收说明

- `paddle.optim.lr_scheduler` 应可用，并公开 `LambdaLR`、`MultiplicativeLR`、`StepLR`、`MultiStepLR`、`ConstantLR`、`LinearLR`、`ExponentialLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`、`CyclicLR`、`CosineAnnealingWarmRestarts`、`OneCycleLR` 和 `LRScheduler`。
- 这些兼容名称应对应现有 `paddle.optimizer.lr` scheduler 的既有行为，而不是形成不同的 scheduler 实现。
- 原有 `paddle.optimizer.lr` namespace 和有效 scheduler 行为必须保持不变。

## 技术要求

- Python package / module import 机制
- PaddlePaddle optimizer 与 learning-rate scheduler API
- API compatibility 与 regression testing

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/79035

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

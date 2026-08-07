# 为 Optimizer `step` API 增加 `closure` 参数支持

## 详细描述

目前 Paddle 的 Optimizer 通过 `step()` 执行一次参数更新，但 `step()` 不能接收 `closure` 参数。在迁移使用 `optimizer.step(closure)` 的训练代码时，这类调用会因为参数不兼容而无法直接运行。

希望为 Optimizer 的 `step` API 增加可选的 `closure` 参数。传入 `closure` 时，应执行该函数并返回它计算得到的 loss，同时继续使用本次计算得到的梯度完成参数更新；不传入 `closure` 时，原有的参数更新行为应保持不变。

该行为需要在基础 `Optimizer` 以及 `Adam`、`AdamW` 中保持一致。

## 验收说明

- `Optimizer.step()`、`Adam.step()` 和 `AdamW.step()` 应支持可选的 `closure` 参数，并在传入时返回 `closure` 计算得到的 loss。
- 执行 `closure` 时应允许正常进行梯度计算，并使用本次计算得到的梯度完成后续参数更新。
- 不传入 `closure` 时，现有 `step()` 的参数更新行为和返回值应保持不变。

## 技术要求

- 熟悉 Python 可调用对象和函数参数设计
- 熟悉 Paddle Optimizer API 与动态图梯度计算
- 能够为 Optimizer 参数更新流程补充回归测试

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/78570

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

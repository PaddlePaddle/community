# 完善 paddle.cdist 对零尺寸 Tensor 的支持

## 详细描述

完善 `paddle.cdist` 对零尺寸 Tensor 的支持，覆盖高维 batch 输入及动态图求导场景。

## 验收说明

- 零尺寸输入下，输出 shape 应符合 batch 维度广播规则
- 正确处理不同 `stop_gradient` 状态的输入组合
- 支持对零尺寸输出执行允许 unused gradient 的求导流程
- 在对应单测中增加高维 batch、可求导和不可求导场景
- 非零尺寸输入的现有行为保持不变

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape 广播规则
- 了解 Paddle 动态图自动微分机制
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- The behavior described above should be supported correctly.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

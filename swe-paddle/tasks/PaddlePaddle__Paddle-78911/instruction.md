# 完善 recompute 的 context detection

## 详细描述

完善动态图 recompute 的 context detection，使 `is_in_recompute()` 在 forward execution 和 backward recomputation 期间均能正确识别 recompute context。

## 验收说明

- 使用 recompute 时，计算函数在首次前向执行期间应处于 recompute 上下文
- 反向传播触发重新计算时，计算函数仍应处于 recompute 上下文
- 启用或关闭 RNG 状态保存时，上述行为均应保持一致
- recompute 执行结束后，context state 应被正确清理
- 现有 gradient computation 和 recompute behavior 不得退化

## 技术要求

- 熟悉 Python 和 Paddle 动态图
- 了解自动微分与 activation recompute 流程
- 了解 Python 上下文管理器

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

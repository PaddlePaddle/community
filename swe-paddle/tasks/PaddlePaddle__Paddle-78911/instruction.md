# 完善 recompute 执行阶段检测

## 详细描述

完善动态图 reentrant recompute 流程中的执行阶段检测，使计算函数在前向执行和反向重计算期间都能通过 `is_in_recompute()` 获得一致、准确的状态。

## 验收说明

- 使用 reentrant recompute 时，计算函数在首次前向执行期间应处于 recompute 上下文
- 反向传播触发重新计算时，计算函数仍应处于 recompute 上下文
- 启用或关闭 RNG 状态保存时，上述行为均应保持一致
- recompute 执行结束后，上下文状态应被正确清理
- 现有梯度计算和 recompute 行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle 动态图
- 了解自动微分与 activation recompute 流程
- 了解 Python 上下文管理器

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

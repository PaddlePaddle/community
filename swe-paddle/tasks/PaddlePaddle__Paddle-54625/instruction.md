# 修复 pipeline parallel 中间输出的不安全释放

## 详细描述

修复 pipeline parallel 在发送中间输出后回收 Tensor 底层数据时的不安全释放问题。未初始化的 Tensor 或已经发生 in-place 修改的 Tensor 不应被清理，否则可能破坏后续访问或 autograd 生命周期；满足安全条件的正常中间输出仍应按原有行为释放。

## 验收说明

- 未初始化的 pipeline 中间输出 Tensor 不应被释放
- 已发生 in-place 修改的 pipeline 中间输出 Tensor 不应被释放
- 已初始化且未发生 in-place 修改的 Tensor，以及 tuple/list 输出的既有释放行为必须保持不变

## 技术要求

- 熟悉 Python
- 熟悉 Paddle dynamic graph 和 Tensor lifecycle
- 了解 pipeline parallel 中间输出的发送、缓存与释放流程

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

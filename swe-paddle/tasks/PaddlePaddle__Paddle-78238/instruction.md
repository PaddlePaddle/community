# 完善 put_along_axis 对零尺寸索引 Tensor 的支持

## 详细描述

完善 `paddle.put_along_axis` 及其原地变体对零尺寸索引 Tensor 的支持。

## 验收说明

- 当索引 Tensor 包含零长度维度时，接口应正常完成
- 非原地接口应返回与输入 Tensor shape 和数据一致的结果
- 原地接口应保持输入 Tensor 的 shape 和数据不变
- 应覆盖 `values` 与零尺寸索引 shape 不一致但无需执行更新的场景
- 非零尺寸索引的现有行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape 与广播规则
- 了解原地与非原地算子的行为差异
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- The behavior described above should be supported correctly.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

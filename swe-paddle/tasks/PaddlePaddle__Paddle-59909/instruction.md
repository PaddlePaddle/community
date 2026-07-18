# 修复 hybrid parallel topology 的 parallel mode 选择

## 详细描述

当 tensor parallel 和 pipeline parallel 均未启用，而 sharding 与 data parallel 同时启用时，hybrid parallel topology 可能错误选择 `DATA_PARALLEL`，导致后续逻辑无法按照实际启用的 sharding strategy 运行。

## 验收说明

- sharding degree 大于 1 时，即使同时启用 data parallel，也应选择 `SHARDING_PARALLEL`
- 仅启用 data parallel 时应继续选择 `DATA_PARALLEL`
- tensor parallel 和 pipeline parallel 的现有 mode selection 不得退化

## 技术要求

- 熟悉 Python
- 了解 Paddle hybrid parallel topology
- 了解 data parallel、tensor parallel、pipeline parallel 和 sharding

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

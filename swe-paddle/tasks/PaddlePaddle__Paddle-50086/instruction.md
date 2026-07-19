# 修复 ConditionalBlock 重复执行时的 scope lifecycle

## 详细描述

`ConditionalBlock` 可能在同一个 operator instance 上重复执行。外层 executor 的生命周期变化可能使上一次执行使用的 child scope 不再有效，但后续执行仍可能继续使用旧 scope，从而访问失效的 execution state。

## 验收说明

- 每次需要执行 sub-block 时都应获得当前有效且独立的 child scope
- 外层 scope 的 children 被清理或替换后，不得继续使用此前的失效 scope
- 首次执行以及 legacy executor 下的现有行为不得退化

## 技术要求

- 熟悉 C++
- 了解 Paddle control-flow operator 和 Scope lifecycle
- 了解 pointer ownership、cached runtime state 和 executor reuse

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

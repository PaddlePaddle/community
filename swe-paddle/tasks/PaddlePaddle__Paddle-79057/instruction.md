# 修复 RestrictedUnpickler 的继承类安全校验绕过

## 详细描述

`RestrictedUnpickler` 的 safe-class check 可能无法识别通过 MRO 继承的 unsafe pickle hooks，导致本应被拒绝的 user-defined class 进入 restricted unpickling path。需要修复该问题，确保 unsafe user-defined restoration logic 在执行前被阻止，同时保持仅依赖默认 `object` behavior 的普通 user-defined class 可用。

## 验收说明

- 通过继承获得不安全序列化或状态恢复行为的类必须被拒绝
- restricted unpickling path 必须在执行不安全的用户自定义恢复逻辑前终止
- 直接声明不安全行为的类应继续被拒绝
- 仅使用 Python 默认对象行为的普通用户自定义类应保持可用
- 不应通过全面禁止用户自定义类来规避该问题
- 现有安全反序列化行为不得退化

## 技术要求

- 熟悉 Python pickle 反序列化机制
- 了解 Python 类继承
- 了解反序列化安全风险
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- Unsafe behavior inherited from a parent class cannot bypass restricted unpickling checks.
- Unsafe user-defined restoration logic is blocked before execution.
- Existing safe user-defined classes remain supported.
- The fix must not weaken tests or broadly disable user-defined classes.

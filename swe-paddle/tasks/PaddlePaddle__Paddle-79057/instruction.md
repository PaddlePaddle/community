# 修复 RestrictedUnpickler 的继承类安全校验绕过

## 详细描述

Paddle 的受限反序列化机制在判断用户自定义类是否安全时，可能无法识别通过继承获得的自定义序列化或状态恢复行为。这会导致部分本应被拒绝的类型被错误接受，并进入受限反序列化流程。需要修复该安全校验问题，使类型的实际反序列化行为能够被正确识别，同时保持普通用户自定义类的现有兼容性。

## 验收说明

- 通过继承获得不安全序列化或状态恢复行为的类必须被拒绝
- 受限反序列化流程必须在执行不安全的用户自定义恢复逻辑前终止
- 直接声明不安全行为的类应继续被拒绝
- 仅使用 Python 默认对象行为的普通用户自定义类应保持可用
- 不应通过全面禁止用户自定义类来规避该问题
- 现有安全反序列化行为不得退化

## 技术要求

- 熟悉 Python pickle 反序列化机制
- 了解 Python 类继承和特殊方法查找规则
- 了解反序列化安全风险
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- Unsafe behavior inherited from a parent class cannot bypass restricted unpickling checks.
- Unsafe user-defined restoration logic is blocked before execution.
- Existing safe user-defined classes remain supported.
- The fix must not weaken tests or broadly disable user-defined classes.

# 支持 SOT 中的 Python 内置 super()

## 详细描述

当 dynamic-to-static 程序在类方法中使用 Python 内置 `super()` 时，SOT 无法稳定保持 eager execution 的继承解析和绑定语义。无参数与显式参数形式、单继承与多继承、父类属性访问以及将 `super` 对象作为函数输入等场景可能触发 breakgraph 或产生不一致行为，从而使包含常见面向对象代码的函数无法作为完整图执行。

## 验收说明

- SOT 应在单继承和多继承中正确执行无参数及显式参数形式的 `super()` 方法调用，并与 eager execution 的结果一致。
- 父类属性访问、`super` 对象作为输入、非标准实例参数名和 guard/cache 场景应保持正确语义且不触发 breakgraph。
- 自定义同名 `super` 函数以及已有对象调用、属性查找和方法绑定行为应保持兼容。

## 技术要求

- Python 对象模型、MRO 与 descriptor/method binding 语义
- Paddle SOT / dynamic-to-static 执行机制
- Python 字节码版本差异、变量追踪与 guard 调试

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/71865

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

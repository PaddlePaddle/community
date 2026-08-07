# 修复 `ConditionalBlock` 复用失效 `Scope` 的问题

## 详细描述

启用 `FLAGS_control_flow_use_new_executor` 后，`ConditionalBlock` 会缓存子块执行时使用的 `Scope`。

外层执行器可能已经释放这个 `Scope`。再次运行同一个 `ConditionalBlock` 时，如果继续使用之前缓存的对象，会访问到已经失效的 `Scope`，导致执行异常。

`ConditionalBlock` 再次运行时应使用仍然有效的 `Scope`，不能复用已经被释放的对象。

## 验收说明

* 重复运行同一个 `ConditionalBlock` 时，不应使用已经失效的 `Scope`
* 外层 `Scope` 被释放后，再次运行 `ConditionalBlock` 不应报错
* 首次运行 `ConditionalBlock` 的行为保持不变
* `ConditionalBlock` 原有的计算结果保持不变

## 技术要求

* 熟悉 C++
* 了解 Paddle 的 `ConditionalBlock`
* 了解 `Scope` 的创建和生命周期
* 了解 `InterpreterCore`

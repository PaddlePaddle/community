# 修复 normal 单元测试的执行模式依赖

## 详细描述

`normal` 算子的部分单元测试依赖调用前遗留的全局 graph mode，导致测试结果受执行顺序影响。 本 issue 中，**graph mode isolation** 是指静态图和动态图测试路径应显式进入各自所需的执行模式，并在结束后恢复调用前的模式状态。`program_guard` 仅用于切换当前的 `main Program` 和 `startup Program`，不能替代 `paddle.enable_static()` 或 `paddle.disable_static()` 完成 graph mode 切换。 需要修复相关测试，使 scalar 参数、Tensor 参数以及 real 和 complex dtype 场景均可独立运行，不依赖其他测试遗留的全局状态。

## 验收说明

- 静态测试路径在调用前处于动态图模式时仍应能够正确构图和执行
- 标量参数和 Tensor 参数场景都应得到覆盖
- 静态测试结束后不得破坏后续动态图测试
- 现有动态图测试行为应保持不变
- 实数和复数测试结构应保持一致的模式隔离能力
- 不得通过删除核心用例、弱化数值断言或跳过测试来规避失败

## 技术要求

- 熟悉 Paddle 静态图和动态图执行模式
- 了解 `Program`、`program_guard`、`Executor` 和静态输入
- 了解 Python 单元测试中的全局状态隔离
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- Static test helpers do not depend on the caller's current graph mode.
- Scalar and tensor-parameter cases execute successfully.
- Existing dygraph behavior remains valid.
- Test ordering no longer determines whether the affected cases pass.
- The fix does not remove or weaken the substantive normal-distribution checks.

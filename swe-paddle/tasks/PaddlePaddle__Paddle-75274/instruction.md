# 修复 normal 单元测试的执行模式依赖

## 详细描述

`normal` 算子的部分单元测试依赖进程当前的静态图或动态图状态。当这些测试以不同顺序运行，或在其他测试改变全局执行模式后单独调用时，静态测试路径可能在错误的模式下创建输入和输出，进而出现构图、数据输入或执行器取值错误。 需要修复该测试实现，使静态和动态图测试路径能够独立运行，不依赖调用前遗留的全局模式状态。

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

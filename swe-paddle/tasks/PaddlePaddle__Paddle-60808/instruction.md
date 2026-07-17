# 修复 broadcast_to 对 0-D Tensor shape dimension 的支持

## 详细描述

在 static graph 或 dynamic-to-static 场景中，当 `paddle.broadcast_to` 的 `shape` 由 Python integer 和 0-D integer Tensor 混合组成时，合法调用会在 graph construction 阶段失败。 例如，部分 output dimension 由其他 Tensor 的 shape 计算得到，并作为 0-D Tensor 放入 `shape` list。此时 `broadcast_to` 应当能够使用这些 dimension 构建并执行计算，而不是拒绝该输入。

## 验收说明

- `broadcast_to` 应支持由 Python integer 和 0-D integer Tensor 混合组成的 `shape` list
- static graph 下应能够成功完成 graph construction 和 execution
- dynamic-to-static 转换后的等价调用应保持可用
- 返回 Tensor 的 shape 和 broadcasted values 应符合请求的目标 shape
- 使用完整 1-D integer Tensor 表示 target shape 的现有行为不得退化
- 非法 shape rank、dtype 或 element type 仍应触发合理的输入校验错误

## 技术要求

- 熟悉 Python
- 了解 Paddle static graph 和 dynamic-to-static 执行流程
- 了解 Tensor shape representation 和 broadcasting semantics
- 了解 Paddle Python API 单元测试开发流程

## Acceptance Criteria

- A valid shape list containing 0-D Tensor dimensions is accepted by `broadcast_to`.
- Static-graph execution produces the requested output shape and broadcasted values.
- Existing integer-list and 1-D shape Tensor behavior remains unchanged.
- Invalid shape inputs continue to be rejected.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

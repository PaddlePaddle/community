# 补齐多 Shape 广播 API

## 详细描述

Paddle 需要提供一个公开的 `broadcast_shapes` API，用于根据常规 broadcasting 规则计算零个、一个或多个 shape 的共同结果。当前已有的二元 `broadcast_shape` 只能处理两个 shape，调用方无法通过对应公共 API 一次处理更长的 shape 序列。

新 API 应正确处理多个可广播 shape、空输入和单 shape 输入，并在输入之间无法广播时保持明确的错误行为。已有的 `broadcast_shape` 行为需要保持兼容。

## 验收说明

- `paddle.broadcast_shapes` 应能够接受多个 shape 并返回它们共同的 broadcasted shape，不兼容输入应失败而不是返回错误结果。
- 零个 shape、单个 shape以及 empty shape 参与广播时应具有合理且稳定的 identity 行为。
- 已有 `paddle.broadcast_shape` 的有效输入、返回值和错误行为应保持不变。

## 技术要求

- 熟悉 Tensor broadcasting 规则与 shape 推导语义。
- 理解 Paddle Python API 的公共导出和 tensor math API 组织方式。
- 测试应验证返回值、异常和兼容行为，不依赖源码字符串、局部变量名或具体循环实现。

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/74594

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid broadcasting behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or broadly bypassing shape validation.

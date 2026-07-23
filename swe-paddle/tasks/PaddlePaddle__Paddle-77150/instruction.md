# 修复部分不可导输出的 PyLayer 在 paddle.grad 中异常退出的问题

## 详细描述

Paddle 自定义 `PyLayer` 可以返回多个值，其中 Tensor 输出既可以参与梯度计算，也可以通过 `mark_non_differentiable` 标记为不可导，并且返回值中可以混合非 Tensor 对象。

当一个 `PyLayer` 同时返回可导 Tensor、不可导 Tensor和非 Tensor 值，并将 Tensor 输出组合后通过 `paddle.grad` 对输入求梯度时，当前实现可能异常终止测试进程，而不是返回正确梯度。

## 验收说明

- `paddle.grad` 应正常完成，不发生段错误或其他进程异常退出。
- 返回梯度应与语义等价的原生 Paddle 计算图一致：不可导输出不贡献梯度，可导输出正常参与反向传播。
- 返回值中混合非 Tensor 对象时，反向传播仍应正确。
- 已有多输出 `PyLayer` 通过 `Tensor.backward()` 执行的行为和梯度结果应保持不变。

## 技术要求

- 熟悉 Python、C++ 和 Paddle Eager autograd。
- 了解自定义 `PyLayer` 的前向、反向和梯度节点生命周期。
- 修复应局限于必要的节点状态传播，不应通过禁用梯度、吞掉异常或改变所有输出的可导状态来规避问题。

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid multi-output `PyLayer` behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, swallowing exceptions, disabling gradients, or bypassing `paddle.grad`.

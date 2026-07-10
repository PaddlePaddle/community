# 将 paddle.inverse 下沉到 C++ 并增强 API 兼容性

## 详细描述

将 `paddle.inverse` 下沉（sink）到 C++ 实现，并在下沉过程中完善其 API 兼容性，使其与 PyTorch 的 `torch.inverse` 对齐，同时保证 `paddle` 提供的多条访问路径行为一致。

需要达成的目标：

- `paddle.inverse` 支持位置参数与关键字参数 `x`，并新增参数别名 `input`（即 `paddle.inverse(input=x)` 与 `paddle.inverse(x=x)` 等价，对齐 PyTorch）。
- `paddle.inverse` 新增 `out` 参数，支持将结果写入用户传入的输出 Tensor（如 `paddle.inverse(x, out=out)`）。
- 保留并兼容 `name` 参数。
- 下沉后，以下三条访问路径行为保持一致，且都指向下沉后的实现：
  - `paddle.inverse`
  - `paddle.Tensor.inverse`（即 `x.inverse()`）
  - `paddle.linalg.inv`
- 下沉不改变 inverse 的数值语义：动态图与静态图结果一致，支持 float32/float64/complex64/complex128，含批量矩阵与复数梯度。

## 验收说明

- 上述所有目标行为均可用，且相关兼容性用例通过。
- 已有的 inverse 语义（数值、梯度、动态图/静态图、各 dtype）保持不变。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle 公开 API 的实现与暴露方式

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

# 为 dsplit、hsplit 和 vsplit 补充参数别名兼容

## 详细描述

在迁移使用 PyTorch 参数命名方式的代码时，`paddle.hsplit`、`paddle.dsplit` 和 `paddle.vsplit` 仍只接受 Paddle 原有的 `x` 与 `num_or_indices` 参数名。使用 `input` 传入待切分 Tensor，或使用 `indices` / `sections` 指定切分位置或份数时，会因为参数名不被识别而调用失败。

需要让这三个 API 在保持现有调用方式不变的同时，接受对应的兼容参数名，并保证通过别名调用得到的切分结果与现有参数名调用一致。

## 验收说明

验证 `hsplit`、`dsplit` 和 `vsplit` 均支持：

- `input` 作为 `x` 的兼容参数名；
- `indices` 和 `sections` 作为 `num_or_indices` 的兼容参数名；
- 原有位置参数以及 `x` / `num_or_indices` 关键字调用继续正常工作。

## 技术要求

- 不改变三个 API 现有切分语义和返回结果。
- 不要求修改底层 C++/CUDA kernel。
- 不应通过放宽全局参数校验或绕过调用检查来实现兼容。

## 参考资料

- 来源 issue: https://github.com/PaddlePaddle/Paddle/issues/76301
- 来源 PR: https://github.com/PaddlePaddle/Paddle/pull/78048

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

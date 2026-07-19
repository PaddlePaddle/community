# 修复 static graph 下 bmm 对 dynamic shape 的支持

## 详细描述

在 static graph 中，`paddle.bmm` 的输入 shape 可能包含值为 `-1` 的 unknown dimension。当两个三维输入在 batch dimension 或 matrix inner dimension 中只有一侧为 unknown，而另一侧为已知值时，合法的 graph construction 可能被错误拒绝，或 output shape 无法正确使用已知 dimension。

## 验收说明

- unknown dimension 不应被当作已知的不匹配 dimension 拒绝
- 能够确定的 output batch、row 和 column dimensions 应被正确保留
- 已知且不兼容的 input shapes 仍应报错，known-shape 行为不得退化

## 技术要求

- 熟悉 Python 和 C++
- 了解 Paddle static graph 与 infermeta
- 了解 batched matrix multiplication 和 dynamic shape

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

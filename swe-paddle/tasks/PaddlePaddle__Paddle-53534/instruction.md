# 修复 static graph 中 to_tensor 的 NumPy 兼容性

## 详细描述

在 static graph 中，`to_tensor` 接收包含 Tensor/Variable 的 list 或 tuple 时，较新 NumPy 版本可能在构造 array 阶段直接报错，而不是返回 object array。这会导致原本可转换的 nested sequence 无法生成 Tensor。

对于 dict 等不支持的输入类型，当前流程也可能在较晚阶段产生不清晰或不稳定的异常。

## 验收说明

- 包含 Tensor/Variable 的 nested sequence 在 NumPy 1.24+ semantics 下应正常转换
- 转换结果应保留预期 value、dtype 和 `stop_gradient`
- 普通 numeric sequence、已有 Variable 和显式 dtype conversion 的行为不得退化
- unsupported input type 应给出明确且稳定的错误信息

## 技术要求

- 熟悉 Python 和 NumPy
- 了解 Paddle static graph 与 Tensor/Variable
- 了解 dtype conversion 和 nested sequence handling

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

# 新增 sparse.mask_as API

## 详细描述

为 Paddle 稀疏计算新增 `sparse.mask_as` API。该 API 根据给定的稀疏 mask（SparseCooTensor 或 SparseCsrTensor），从稠密 Tensor 中提取对应非零位置的值，输出一个与 mask 具有相同 indices 的稀疏 Tensor。

要求支持：
- COO 格式：支持 1-D ~ 4-D 输入
- CSR 格式：支持 2-D 和 3-D 输入（其他维度应报错）
- 数据类型：float32, float64, int32, int64, complex64, complex128, int8, int16, float16
- 前向计算 + 反向梯度

## 验收说明

- `paddle.sparse.mask_as(x, mask)` 应能正确根据 mask 的 indices 从稠密 Tensor x 中提取值
- 支持 COO 和 CSR 两种稀疏格式
- CSR 格式仅支持 2-D 和 3-D，其他维度应报错
- 反向梯度应正确传播

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

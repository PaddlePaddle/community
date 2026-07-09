# 为稀疏计算 multiply/divide 添加复数支持

## 详细描述

为以下稀疏计算添加 complex64/complex128 数据类型支持：multiply_coo_coo、multiply_csr_csr、divide_coo_coo、divide_csr_csr

## 验收说明

- 在 op 对应的前向以及反向 kernel 增加复数运算逻辑，且注册相应的 complex64, complex128 数据类型
- 在对应 op 的单测中增加复数类型
- 在对应 api 的类型校验中增加复数

## 技术要求

- 熟悉 Python、C++，CUDA
- 了解 Paddle 算子开发流程
- 了解稀疏矩阵的表示方式

## 参考资料

- https://github.com/PaddlePaddle/Paddle/issues/61975

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

# paddle_lu_solve 设计文档

| API 名称     | paddle.lu_solve                     |
| ------------ | ----------------------------------- |
| 提交作者     | Chen-Lun-Hao                        |
| 提交时间     | 2024-04-15                          |
| 版本号       | V2.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名       | 20240415_api_design_for_lu_solve.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，需要为飞桨扩充 API `paddle.lu_solve`

本 API 属于飞桨开源个人贡献赛 API 开发任务[No.12：为 Paddle 新增 lu_solve API](https://github.com/PaddlePaddle/Paddle/issues/62905)的任务。

## 2、功能目标

使用 LU 分解 来求解线性方程组 AX=B，A 为 1 个或多个矩阵，A.shape=[m, n] or [batch, m, n]，B.shape=[m, k] or [batch, m, k]，A 和 B 已知，通过 LU 分解方阵 A 来加速求解 X。需要满足 LU, pivots, info =paddle.linalg.lu(A); X = paddle.linalg.lu_solve(B, LU, pivots) 与 使用 X=paddle.linalg.solve(A, B) 直接求解线性方程组的结果一样。

预期该 API 支持

- paddle.linalg.lu_solve 作为独立的函数调用
- Tensor.lu_solve 作为 Tensor 的方法使用

## 3、意义

为飞桨增加求解线性方程组 AX=B 的计算方式，提升飞桨 API 丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) → Tensor` 以及对应的 `torch.lu_solve`

其介绍为：

> Computes the solution of a square system of linear equations with a unique solution given an LU decomposition.

Letting K be R or C, this function computes the solution X∈K^(n×k) of the linear system associated to
A∈K^(n×n),B∈K^(n×k), which is defined as AX=B
where A is given factorized as returned by lu_factor().

> 参数表为：

- `Tensor` Lu: tensor of shape (_, n, n) where _ is zero or more batch dimensions.
- `Tensor` pivots: tensor of shape (_, n) where _ is zero or more batch dimensions.
- `Tensor` B: right-hand side tensor of shape (\*, n, k).
- `Optional[bool]` left: whether to solve the system **A**X=**B or **X**A**=B. Default: True.
- `Optional[bool]` adjoint: whether to solve the system AX=B or A^**H**X=B. Default: False.
- `Tensor` out: output tensor. Ignored if None. Default: None.

### 实现

PyTorch 在 2.2 版本给出的 API 中，针对 `lu_solve`操作进行实现的代码如下，具体代码可以参考[BatchLinearAlgebraKernel.cpp](https://github.com/pytorch/pytorch/blob/99c822c0ba747fad8528ff6b57712abdbdc2c093/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L2710)

```cpp
void apply_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  auto b_data = B.data_ptr<scalar_t>();
  auto lu_data = LU.const_data_ptr<scalar_t>();
  const auto trans = to_blas(transpose);
  auto pivots_data = pivots.const_data_ptr<int>();
  auto b_stride = matrixStride(B);
  auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;
  auto batch_size = batchCount(B);

  auto n = LU.size(-2);
  auto nrhs = B.size(-1);
  auto leading_dimension = std::max<int64_t>(1, n);

  int info = 0;

  // lu and pivots tensors can be broadcast to B
  // here we construct a helper indexing tensor to linearly index into LU and pivots
  IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
  IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
  BroadcastLinearIndices lu_index(
      batchCount(LU), lu_batch_shape, b_batch_shape);

  for (const auto i : c10::irange(batch_size)) {
    int64_t lu_index_i = lu_index(i);
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    const scalar_t* lu_working_ptr = &lu_data[lu_index_i * lu_stride];
    const int* pivots_working_ptr = &pivots_data[lu_index_i * pivots_stride];

    lapackLuSolve<scalar_t>(trans, n, nrhs, const_cast<scalar_t*>(lu_working_ptr), leading_dimension, const_cast<int*>(pivots_working_ptr),
                            b_working_ptr, leading_dimension, &info);

    // info from lapackLuSolve only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_lu_solve'
void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Lapack will write into unrelated memory if pivots are not in the right range so we do
  // some simple sanity checks here for the CPU version
  TORCH_CHECK(pivots.gt(0).all().item<bool>(),
              "Pivots given to lu_solve must all be greater or equal to 1. "
              "Did you properly pass the result of lu_factor?");
  TORCH_CHECK(pivots.le(LU.size(-2)).all().item<bool>(),
              "Pivots given to lu_solve must all be smaller or equal to LU.size(-2). "
              "Did you properly pass the result of lu_factor?");

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "linalg.lu_solve_cpu", [&]{
    apply_lu_solve<scalar_t>(LU, pivots, B, trans);
  });
}
```

# 四、算子分析

PyTorch 通过 c++ 实现，在底层封装 lapack 库，为 python 提供调用接口。

# 五、设计思路与实现方案

可通过设计底层 op 来调用 lapack 中的相关代码实现 lu_solve。在 python/paddle 中实现对数据的类型，维度等检查，并在底层实现实际的操作方法。

## 命名与参数设计

添加 Python API:

```python
paddle.lu_solve(lu, pivots, b, name=None)
```

参数表：

- lu: (Tensor) LU 分解的结果。shape 为 （_， m， m） ，其中 _ 是 batch 的维度, 数据类型为 float32、float64。
- pivots: (Tensor) LU 分解的主元。shape 为 （_， m） ，其中 _ 是 batch 的维度， 数据类型为 int32。
- b: (tuple) 列向量 b 。 b 的 shape 为 （_， m， k） ，其中 _ 是 batch 的维度, 数据类型为 float32、float64
- name: (Optional[str]) op 名称

## 底层 OP 设计

涉及底层 OP，在 paddle/phi 中定义模板函数

```cpp
// LU_solve
template <typename T>
void lapackLuSolve(int n, int nrhs, T *a, int lda, int *ipiv, T \*b, int ldb);
```

# 六、测试和验收的考量

- 覆盖动态图和静态图的测试场景
- 覆盖 CPU、GPU 两种测试场景
- 支持 Tensor 精度 FP32、FP64
- 需要检查计算正确性
- 需要检查二到三维的情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关 PythonAPI 均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

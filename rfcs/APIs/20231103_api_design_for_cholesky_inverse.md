# paddle.cholesky_inverse 设计文档

| API名称      | Paddle.cholesky_inverse                     |
| ------------ | ------------------------------------------- |
| 提交作者     | kafaichan                                        |
| 提交时间     | 2023-11-03                                  |
| 版本号       | V1.0                                        |
| 依赖飞桨版本 | develop                                     |
| 文件名       | 20231103_api_design_for_cholesky_inverse.md |


# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，支持矩阵计算类API，Paddle需要扩充API `paddle.linalg.cholesky_inverse`。 该功能实现使用Cholesky因子矩阵u计算一个半正定矩阵的逆矩阵。

## 2、功能目标
实现`cholesky_inverse` API,使用 Cholesky 因子U计算对称正定矩阵的逆矩阵，调用方式如下:
 - paddle.cholesky_inverse , 作为独立的函数调用
 - Tensor.cholesky_inverse , 作为 Tensor 的方法使用

## 3、意义
飞桨支持API `paddle.linalg.cholesky_inverse`

# 二、飞桨现状
目前paddle缺少相关功能实现。

无类似功能API或者可组合实现方案。


# 三、业内方案调研

**pytorch**

API为：`torch.cholesky_inverse`(*input*, *upper=False*, *, *out=None*) → Tensor

PyTorch参数的*用于指定后面参数out为keyword-only参数，Paddle中无此参数

# 四、对比分析

  - PyTorch是在C++ API基础上实现，使用Python调用C++对应的接口
  - Scipy是通过调用lapack的方法实现
  - Tensorflow、Numpy中没有对应API的实现

## Pytorch

### 实现解读

在PyTorch中，cholesky_inverse底层是由C++实现的，
1. GPU端的核心代码为:
```C++
template <typename scalar_t>
static void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(false, "cholesky_inverse: MAGMA library not found in compilation. Please rebuild with MAGMA.");
#else
  // magmaCholeskyInverse (magma_dpotri_gpu) is slow because internally
  // it transfers data several times between GPU and CPU and calls lapack routine on CPU
  // using magmaCholeskySolveBatched is a lot faster
  // note that magmaCholeskySolve is also slow

  // 'input' is modified in-place we need to clone it and replace with a diagonal matrix
  // for apply_cholesky_solve
  auto input_working_copy = cloneBatchedColumnMajor(input);

  // 'input' tensor has to be a batch of diagonal matrix
  input.fill_(0);
  input.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);

  Tensor result_u, input_u;
  if (input.dim() == 2) {
    // unsqueezing here so that the batched version is used
    result_u = input.unsqueeze(0);
    input_u = input_working_copy.unsqueeze(0);
  } else {
    result_u = input;
    input_u = input_working_copy;
  }

  // magma's potrs_batched doesn't take matrix-wise array of ints as an 'info' argument
  // it returns a single 'magma_int_t'
  // if info = 0 the operation is successful, if info = -i, the i-th parameter had an illegal value.
  int64_t info_tmp = 0;
  apply_cholesky_solve<scalar_t>(result_u, input_u, upper, info_tmp);
  infos.fill_(info_tmp);
#endif
}
```
在前序一系列参数检查后，进入`apply_cholesky_inverse`函数，通过调用`apply_cholesky_solve`使用MAGMA例程进行求解。


2. CPU端的核心代码为:
```C++
template <typename scalar_t>
void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "cholesky_inverse: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto batch_size = batchCount(input);
  auto n = input.size(-2);
  auto lda = std::max<int64_t>(1, n);

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* info_working_ptr = &infos_data[i];
    lapackCholeskyInverse<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
    // LAPACK writes to only upper/lower part of the matrix leaving the other side unchanged
    apply_reflect_conj_tri_single<scalar_t>(input_working_ptr, n, lda, upper);
  }
#endif
}
```
在前序一系列参数检查后，进入`apply_cholesky_inverse`函数，根据张量的数据类型调用lapack库的``zpotri_``, ``cpotri_``, ``dpotri_``, ``spotrs_`` 函数求解。


## Scipy

### 实现解读
Scipy 使用wrapper的方式调用lapack库的 ``zpotri_``, ``cpotri_``, ``dpotri_``, ``spotrs_`` 函数求解。

[scipy.linalg.lapack.zpotri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.zpotri.html)

[scipy.linalg.lapack.cpotri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.cpotri.html)

[scipy.linalg.lapack.dpotri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dpotri.html)

[scipy.linalg.lapack.spotri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.spotri.html)


# 五、设计思路与实现方案

## 命名与参数设计
`paddle.cholesky_inverse`(*x*, *upper=False*, *name=None*) → Tensor

使用Cholesky分解因子u,计算对称正定矩阵A的逆矩阵inv

- Parameters
    - **x** (Tensor) - ``paddle.cholesky`` 计算的对称正定矩阵A的Cholesky分解因子，大小为 (*, n, n), 其中 * 为零或多个维度。
    - **upper** (bool, 可选) - 输入x是否是上三角矩阵，True为上三角矩阵，False为下三角矩阵。默认值False。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

- Returns
    - Tensor, 如果upper为True, 返回 $(U^TU)^{-1}$ ，否则返回 $(LL^T)^{-1}$

## 底层OP设计
优先考虑实现与PyTorch结果对齐

正向：采用lapack库实现与PyTorch保持一致，调用`dpotri_`、`spotri_`实现。GPU调用cusolver库的函数实现。

反向：参考`paddle/phi/kernels/impl/inverse_grad_kernel_impl.h`的方式实现`paddle/phi/kernels/impl/cholesky_inverse_grad_kernel_impl.h`

## API实现方案
函数实现路径：在`python/paddle/tensor/linalg.py`中增加`cholesky_inverse`函数

单元测试路径：在`test/legacy_test`目录下增加`test_cholesky_inverse.py`


# 六、测试和验收的考量
1. 结果正确性:
   - 前向计算: `paddle.cholesky_inverse`计算结果与 `torch.cholesky_inverse` 计算结果一致。
   - 反向计算: 参考`test/legacy/test/test_cholesky_op.py`的写法，调用`grad_check`方法检查梯度。
   - 测试无效输入，对角线上的元素有0是否报错
2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

# 七、可行性分析和排期规划
方案实施难度可控，工期上可以满足在当前版本周期内开发完成

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无

# 附件及参考资料
[cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)

[torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch.cholesky_inverse)
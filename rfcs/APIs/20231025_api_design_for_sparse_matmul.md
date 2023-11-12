# paddle.sparse.matmul 增强设计文档

|API名称 | paddle.sparse.matmul | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-10-25 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20231025_api_design_for_sparse_matmul.md<br> | 


# 一、概述
## 1、相关背景
深度学习中有许多模型使用到了稀疏矩阵的乘法算子。稀疏矩阵的乘法一般可支持 `COO*Dense`、`COO*COO`、`CSR*Dense`、`CSR*CSR` 四种计算模式，目前 Paddle 已支持 `COO*Dense`、`CSR*Dense` 两种。


## 2、功能目标
为稀疏 API `paddle.sparse.matmul` 完善 `COO*COO`、`CSR*CSR` 两种计算模式。

## 3、意义

完善稀疏API `paddle.sparse.matmul` 的功能，提升稀疏 tensor API 的完整度。

# 二、飞桨现状
目前 Paddle 已支持 `COO*Dense`、`CSR*Dense` 计算模式，不支持 `COO*COO`、`CSR*CSR` 计算模式。

# 三、业内方案调研
## PyTorch

Pytorch 在计算 `Sparse * Sparse` 时，会将两个相乘的矩阵统一为 `CSR` 模式，再调用 `CSR*CSR` 计算模式的 kernel， 计算得到 `CSR` 计算结果，最后将计算结果转换 `COO` 模式，API 调用方式如下：

```python
torch.sparse.mm
torch.mm
```

CPU 版本的 kernel 为 `sparse_matmul_kernel` ，其代码的位置在 `pytorch/aten/src/ATen/native/sparse/SparseMatMul.cpp`。

CPU kernel 实现了 [Sparse matrix multiplication package (SMMP)](https://doi.org/10.1007/BF02070824) 论文中的稀疏矩阵乘法算法。

GPU 版本的 kernel 为 `sparse_sparse_matmul_cuda_kernel` ，其代码的位置在 `pytorch/aten/src/ATen/native/sparse/cuda/SparseMatMul.cu`。

GPU kernel 使用了 cuda 实现计算，当 `cudaSparse` 库可用时，kernel 调用 `cusparseSpGEMM` 进行矩阵运算，否则会调用 `thrust` 进行矩阵运算。

## TensorFlow
TensorFlow 中的 `SparseTensor` 使用了 `COO` 模式存储稀疏矩阵，为了支持 `CSR` 模式的稀疏矩阵，专门设计了 `CSRSparseMatrix`。

TensorFlow 中没有 `COO*COO` 计算模式的直接实现。

`CSR*CSR` 计算模式的 API 调用方式如下：
```python
tensorflow.python.ops.linalg.sparse.sparse_matrix_sparse_mat_mul
```

算子实现代码的位置在 `tensorflow\core\kernels\sparse\sparse_mat_mul_op.cc`，包含 CPU 版本的 `CSRSparseMatMulCPUOp` 和 GPU 版本的 `CSRSparseMatMulGPUOp`。

`CSRSparseMatMulCPUOp` 使用了 `Eigen` 库实现计算；`CSRSparseMatMulGPUOp`使用了 `cudaSparse` 库的 `cusparseSpGEMM` 实现计算。


# 四、对比分析
PyTorch 同时支持 `CSR*CSR`、`COO*COO` 计算模式，其底层只实现了 `CSR*CSR` 模式的稀疏矩阵乘法计算，在计算 `COO*COO` 模式时，底层代码会进行稀疏矩阵模式的转换。

TensorFlow 只支持 `CSR*CSR` 计算模式。

PyTorch 和 TensorFlow 在使用 GPU 进行稀疏矩阵乘法计算时，都调用了 `cudaSparse` 库。

Paddle 中已经有部分的稀疏矩阵乘法 API，代码位置在 `paddle/phi/kernels/sparse/gpu/matmul_kernel.cu`，主要是通过`cudaSparse` 库完成计算。

# 五、设计思路与实现方案

## 命名与参数设计
Paddle 中已完成 `COO*COO`、`CSR*CSR` 计算模式的函数设计。
```yaml
- op: matmul
  args : (Tensor x, Tensor y)
  output : Tensor(out)
  infer_meta :
    func : MatmulInferMeta
    param: [x, y, false, false]
  kernel :
    func : matmul_csr_dense {sparse_csr, dense -> dense},
           matmul_csr_csr {sparse_csr, sparse_csr -> sparse_csr},
           matmul_coo_dense {sparse_coo, dense -> dense},
           matmul_coo_coo {sparse_coo, sparse_coo -> sparse_coo}
    layout : x
  backward: matmul_grad
```

## 底层OP设计
Paddle 中已完成 `COO*COO`、`CSR*CSR` 计算模式的 kernel 设计。
```cpp
template <typename T, typename Context>
void MatmulCooCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& y,
                        SparseCooTensor* out);

template <typename T, typename Context>
void MatmulCsrCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& y,
                        SparseCsrTensor* out);
```
对应的反向 kernel。
```cpp
template <typename T, typename Context>
void MatmulCooCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const SparseCooTensor& y,
                            const SparseCooTensor& dout,
                            SparseCooTensor* dx,
                            SparseCooTensor* dy);

template <typename T, typename Context>
void MatmulCsrCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& y,
                            const SparseCsrTensor& dout,
                            SparseCsrTensor* dx,
                            SparseCsrTensor* dy);
```

## API实现方案
在 `paddle/phi/kernels/sparse/gpu/matmul_kernel.cu` 中实现 `MatmulCooCooKernel` 和 `MatmulCsrCsrKernel`；

在 `paddle/phi/kernels/sparse/gpu/matmul_grad_kernel.cu` 中实现 `MatmulCooCooGradKernel` 和 `MatmulCsrCsrGradKernel`。

API 主要通过调用 `cudaSparse` 库完成计算实现，目前暂不需要开发 CPU kernel。

`cudaSparse` 库的 `cusparseSpGEMM` 只支持 `CSR*CSR` 模式，在计算 `COO*COO` 模式时，需要进行 `COO` 和 `CSR` 模式之间的转换。

反向算子计算方式如下：

$dx = dout * y'$

$dy = x' * dout$

# 六、测试和验收的考量

在 `test/legacy_test/test_sparse_matmul_op.py` 中补充对 `COO*COO`、`CSR*CSR` 计算模式的测试。参照 `TestMatmul` 类，测试 2 维和 3 维稀疏矩阵的乘法计算。

# 七、可行性分析和排期规划

## 排期规划
1. 10月25日~10月30日完成 `MatmulCooCooKernel` 和 `MatmulCsrCsrKernel` 的实现。
2. 10月31日~11月4日完成 `MatmulCooCooGradKernel` 和 `MatmulCsrCsrGradKernel` 的实现。
3. 11月5日~11月10日完成测试代码的开发，并编写文档。

# 八、影响面
拓展 `paddle.sparse.matmul` 模块。

# 名词解释
无

# 附件及参考资料

[The API reference guide for cuSPARSE](https://docs.nvidia.com/cuda/cusparse/)
[CSR Sparse Matrix](https://github.com/tensorflow/community/blob/master/rfcs/20200519-csr-sparse-matrix.md)

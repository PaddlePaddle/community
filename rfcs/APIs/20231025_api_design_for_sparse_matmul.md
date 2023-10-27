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
Pytorch 的稀疏矩阵乘法的实现代码的位置在 `pytorch/aten/src/ATen/native/sparse/SparseMatMul.cpp`。

函数 `sparse_matmul_kernel` 在计算稀疏矩阵乘法时，会将两个相乘的矩阵统一为 `CSR` 模式，再进行计算，得到 `CSR` 模式的计算结果。如果想要获得 `COO` 模式的计算结果，需将 `CSR` 模式转化为 `COO` 模式。

```cpp
template <typename scalar_t>
void sparse_matmul_kernel(
    Tensor& output,
    const Tensor& mat1,
    const Tensor& mat2) {
        ...
}
```


## TensorFlow
TensorFlow 的稀疏矩阵乘法的实现代码的位置在 `tensorflow\core\kernels\sparse\sparse_mat_mul_op.cc`。

主要使用了 `cudaSparse` 库的 `cusparseSpGEMM` 实现计算。
```cpp
#if GOOGLE_CUDA && (CUDA_VERSION >= 12000)
    GpuSparse cudaSparse(ctx);
    OP_REQUIRES_OK(ctx, cudaSparse.Initialize());
   
    ...

    OP_REQUIRES_OK(ctx,
                   cudaSparse.SpGEMM_compute<T>(matA, matB, matC, gemmDesc, &bufferSize2, nullptr));
```


# 四、对比分析
PyTorch 中实现了 `CSR*CSR` 模式的稀疏矩阵乘法计算，在其他模式 `CSR*COO`、`COO*COO` 的计算时，会进行稀疏矩阵模式的转换。

TensorFlow 中是调用了 `cudaSparse` 库完成稀疏矩阵乘法计算。

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

# paddle.sparse.mask_as 设计文档

| API名称      | paddle.sparse.mask_as                     |
| ------------ | ----------------------------------------- |
| 提交作者     | megemini                                  |
| 提交时间     | 2024-05-15                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本 | develop版本                               |
| 文件名       | 20240515_api_design_for_sparse_mask_as.md |


# 一、概述

## 1、相关背景

[NO.17 为 Paddle 新增 sparse.mask_as API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no17-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-sparsemask_as-api)

利用稀疏矩阵 `mask` 的 `indices` 过滤输入 Tensor `x`，进而生成对应的稀疏矩阵。

## 2、功能目标

实现 `paddle.sparse.mask_as` 作为独立的函数调。

## 3、意义

新增 `paddle.sparse.mask_as` 方法，丰富 Paddle 稀疏 API。

# 二、飞桨现状

目前没有 `paddle.sparse.mask_as` 接口，但是，sparse 算子中的 `mask_kernel.cc mask_kernel.cu` 实现了相同的功能，需要对其进行封装，并支持 `coo csr` 两种稀疏矩阵，并实现反向算子。

# 三、业内方案调研

PyTorch 中的 [torch.Tensor.sparse_mask](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html#torch-tensor-sparse-mask)

实现了相同的能力。

PyTorch 中 `aten/src/ATen/native/sparse/SparseTensor.cpp`

``` c++
SparseTensor sparse_mask(const Tensor& t, const SparseTensor& mask) {
  TORCH_CHECK(
      mask.sizes().equals(t.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      t.sizes(),
      " but mask has size ",
      mask.sizes());

  if (t.is_same(mask)) {
    return t;
  }

  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(t.device(), t.scalar_type());
  }

  if (t.layout() == at::kSparse) {
    if (!t._nnz()) {
      auto res = mask.clone().to(t.device(), t.scalar_type());
      res._values().zero_();
      return res;
    }

    auto res = at::empty({0}, t.options());
    auto [lhs, rhs, lhs_hash_opt] = sparse_mask_like_prepare_sparse_inputs("sparse_mask", t, mask);
    sparse_mask_intersection_out_stub(res.device().type(), res, lhs, rhs, lhs_hash_opt);
    return res._coalesced_(mask.is_coalesced());
  }

  const auto mask_values = mask._values();
  auto mask_template = at::sparse_coo_tensor(
      mask._indices(),
      at::ones({1}, mask_values.options()).expand_as(mask_values),
      mask.sizes())._coalesced_(mask.is_coalesced());
  return t.mul(mask_template).to(t.scalar_type());
}
```

# 四、对比分析

Paddle 目前在 `paddle/phi/kernels/sparse/cpu/mask_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/mask_kernel.cu` 中实现了 `MaskCooKernel` 算子：

``` c++
void MaskCooKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const SparseCooTensor& mask,
                   SparseCooTensor* out)
```

此算子实现了与 PyTorch 中 `sparse_mask` 相同的能力，但是，Paddle 算子中只有 `coo` 类型的支持，还需要补充 `csr` 类型稀疏矩阵的支持。

另外，Paddle 的 `MaskCooKernel` 并没有暴露至上层 Python 接口，还需要注册此算子，以及涉及对应的反向算子。


# 五、设计思路与实现方案

- 复用  `paddle/phi/kernels/sparse/cpu/mask_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/mask_kernel.cu` 中的 `MaskCooKernel`
- 重新注册算子 `mask_as`
- 实现 `coo` `csr` 两种稀疏矩阵
- 实现反向算子

## 命名与参数设计

``` python
paddle.sparse.mask_as(x: DenseTensor, mask: SparseTensor, name=None)
```

- x (Tensor) - 输入的 DenseTensor
- mask (Tensor) - 利用此 SparseTensor 的 indices 过滤输入
- name (str, optional) - 名称

在 `python/paddle/sparse/binary.py` 中增加 Python 接口

``` python
@dygraph_only
def mask_as(x, mask, name=None):
    return _C_ops.sparse_mask_as(x, mask)
```

> 说明：目前 sparse 中大部分接口都是 `@dygraph_only` ，此接口暂保留此 annotation。

## 底层OP设计

- 注册算子

`paddle/phi/api/yaml/sparse_ops.yaml`

``` yaml
- op: mask_as
  args : (Tensor x, Tensor mask)
  output : Tensor(out)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : mask_as_coo{dense, sparse_coo -> sparse_coo},
           mask_as_csr{dense, sparse_csr -> sparse_csr}
    layout : x
  backward: mask_as_grad
```

`paddle/phi/api/yaml/sparse_backward.yaml`

``` yaml
- backward_op : mask_as_grad
  forward : mask_as(Tensor x, Tensor mask) -> Tensor(out)
  args : (Tensor x, Tensor mask, Tensor out_grad)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : mask_as_coo_grad {dense, sparse_coo, sparse_coo -> dense},
           mask_as_csr_grad {dense, sparse_csr, sparse_csr -> dense}
```

- 实现正向算子

`paddle/phi/kernels/sparse/cpu/mask_kernel.cc`

``` c++
template <typename T, typename Context>
void MaskAsCooKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCooTensor& mask,
                     SparseCooTensor* out) {
  MaskCooKernel<T, Context>(dev_ctx, x, mask, out);
}

template <typename T, typename Context>
void MaskAsCsrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCsrTensor& mask,
                     SparseCsrTensor* out) {
  // transform csr format to coo format, and then use coo kernel
  const SparseCooTensor mask_coo = CsrToCoo<T, Context>(dev_ctx, mask);
  SparseCooTensor out_coo;
  MaskAsCooKernel<T, Context>(dev_ctx, x, mask_coo, &out_coo);
  CooToCsrKernel<T, Context>(dev_ctx, out_coo, out);
}
```

其中，`csr` 类型通过 `CsrToCoo` 转换为 `coo` 后利用 `MaskAsCooKernel` 进行计算。

`paddle/phi/kernels/sparse/gpu/mask_kernel.cu` 逻辑相同

``` c++
template <typename T, typename Context>
void MaskAsCooKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCooTensor& mask,
                     SparseCooTensor* out) {
  MaskCooKernel<T, Context>(dev_ctx, x, mask, out);
}

template <typename T, typename Context>
void MaskAsCsrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const SparseCsrTensor& mask,
                     SparseCsrTensor* out) {
  // transform csr format to coo format, and then use coo kernel
  const SparseCooTensor mask_coo = CsrToCoo<T, Context>(dev_ctx, mask);
  SparseCooTensor out_coo;
  MaskAsCooKernel<T, Context>(dev_ctx, x, mask_coo, &out_coo);
  CooToCsrKernel<T, Context>(dev_ctx, out_coo, out);
}
```

- 实现反向算子

`mask_as` 的反向逻辑为：将输入的 `SparseTensor` 类型的 `grad`，直接转换为 `Dense Tensor`。

`paddle/phi/kernels/sparse/mask_grad_kernel.h`

``` c++
template <typename T, typename Context>
void MaskAsCooGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const SparseCooTensor& mask,
                         const SparseCooTensor& out_grad,
                         DenseTensor* x_grad) {
  CooToDenseKernel<T, Context>(dev_ctx, out_grad, x_grad);
}

template <typename T, typename Context>
void MaskAsCsrGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const SparseCsrTensor& mask,
                         const SparseCsrTensor& out_grad,
                         DenseTensor* x_grad) {
  CsrToDenseKernel<T, Context>(dev_ctx, out_grad, x_grad);
}
```

利用 `CooToDenseKernel` 和 `CsrToDenseKernel` 直接转换 `grad`。

并且，新增反向算子的注册文件:

`paddle/phi/kernels/sparse/cpu/mask_grad_kernel.cc`

``` c++
PD_REGISTER_KERNEL(mask_as_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCooGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_as_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCsrGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
```

`paddle/phi/kernels/sparse/gpu/mask_grad_kernel.cu`

``` c++
PD_REGISTER_KERNEL(mask_as_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCooGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(mask_as_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaskAsCsrGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
```

## API实现方案

# 六、测试和验收的考量

- **编程范式场景**
  - 常规覆盖动态图 (和静态图) 的测试场景。

- **硬件场景**
  - 常规需覆盖 CPU、GPU 两种测试场景。

- **输入参数**
  - 常规覆盖默认参数，常用参数，错误参数。
  - 常规覆盖 `coo` `csr` 两种稀疏矩阵
  - `coo` 测试维度需要大于 `3D`
  - `csr` 测试维度为 `2D` `3D`

- **计算精度**
  - 需要保证 `前向/后向` 计算的精度正确性

# 七、可行性分析和排期规划

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

丰富 paddle API，对其他模块没有影响

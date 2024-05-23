# paddle.sparse.mask_as 设计文档

| API名称      | paddle.sparse.mask_as                     |
| ------------ | ----------------------------------------- |
| 提交作者     | megemini                                  |
| 提交时间     | 2024-05-15                                |
| 版本号       | V1.1                                      |
| 依赖飞桨版本 | develop版本                               |
| 文件名       | 20240515_api_design_for_sparse_mask_as.md |


# 一、概述

## 1、相关背景

目前 Paddle 支持 `coo` 和 `csr` 两种稀疏矩阵格式，需要新增 `mask` 掩码逻辑，利用稀疏矩阵 `mask` 的 `indices` 过滤输入 Tensor `x`，进而生成对应格式的稀疏矩阵。需要新增 2 个 kernel 的前向与反向逻辑，其中 `csr` 的 kernel 需支持 2D/3D Tensor，`coo` 的 kernel 需支持任意维度的 Tensor。

> 参考赛题：[NO.17 为 Paddle 新增 sparse.mask_as API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no17-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-sparsemask_as-api)


## 2、功能目标

实现 `paddle.sparse.mask_as` 作为独立的函数调用。

## 3、意义

新增 `paddle.sparse.mask_as` 方法，丰富 Paddle 稀疏 API。

# 二、飞桨现状

目前没有 `paddle.sparse.mask_as` 接口，但是，sparse 算子中的 `mask_kernel.cc mask_kernel.cu` 实现了相同的功能，需要对其进行封装，支持 `coo csr` 两种稀疏矩阵，并实现反向算子。

# 三、业内方案调研

PyTorch 中的 [torch.Tensor.sparse_mask](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html#torch-tensor-sparse-mask) 实现了相同的能力。

参考 PyTorch 中的 `aten/src/ATen/native/sparse/SparseTensor.cpp`

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

对于 `csr` 格式的矩阵，参考 `aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp` 中的

``` c++
Tensor sparse_mask_sparse_compressed(
    const Tensor& self,
    const Tensor& mask) {
  TORCH_CHECK(at::sparse_csr::is_sparse_compressed(mask),
              "sparse_mask_sparse_compressed expects mask to have sparse compressed layout, got ", mask.layout());
  TORCH_CHECK(
      mask.sizes().equals(self.sizes()),
      "sparse_mask(): operands have incompatible sizes; self has size ",
      self.sizes(),
      " but mask has size ",
      mask.sizes());

  if (self.is_same(mask)) {
    return self;
  }

  if (!mask.numel() || !mask._nnz()) {
    return mask.clone().to(self.device(), self.scalar_type());
  }

  if (self.layout() == kStrided) {
    auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(mask);
    auto mask_values = mask.values();
    auto dense_mask = at::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        at::ones({1}, self.options().dtype(kBool)).expand_as(mask_values),
        self.sizes(),
        self.options().dtype(kBool).layout(mask.layout())).to_dense();
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), {}, mask.dense_dim());
        },
        [&] {
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return at::native::dense_to_sparse_with_mask(self, dense_mask, mask.layout(), blocksize, mask.dense_dim());
        });
  } else if (self.layout() == mask.layout()) {
    // TODO: keeping this for BC but the method used here may lead to
    // incorrect indices.
    return self.mul(at::ones_like(mask)).to(self.scalar_type());
  } else {
    // TODO: keeping this for BC but the method used here cannot
    // support batch dimensions because sparse COO tensors are batch
    // dimension ignorant.
    return AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(
        mask.layout(), "sparse_mask_sparse_compressed",
        [&] {
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout());
        },
        [&] {
          auto blocksize = at::sparse_csr::getBlockSize(mask);
          return self.sparse_mask(mask.to_sparse()).to_sparse(mask.layout(), blocksize);
        });
  }
}
```

其利用 `at::native::dense_to_sparse_with_mask` 处理 `compressed` 格式的稀疏矩阵，包括 `csr` 格式。

具体进行 `dense to sparse` 的逻辑为 `aten/src/ATen/native/TensorConversions.cpp` 中的

``` c++
template<Layout target_layout>
static Tensor dense_to_sparse_compressed(const Tensor& self, const Tensor& self_mask, IntArrayRef blocksize, c10::optional<int64_t> dense_dim_opt) {
  static_assert(target_layout == Layout::SparseCsr || target_layout == Layout::SparseCsc
                || target_layout == Layout::SparseBsr || target_layout == Layout::SparseBsc,
                "invalid layout template parameter for dense_to_sparse_compressed");
  constexpr auto compressed_rows_layout = target_layout == Layout::SparseCsr || target_layout == Layout::SparseBsr;
  constexpr auto blocked_layout = target_layout == Layout::SparseBsr || target_layout == Layout::SparseBsc;

  int64_t dense_dim = dense_dim_opt.value_or(0);

  // Reshape values so that the block dims are explicitly added, and
  // calculate a mask tensor that has only batch and sparse dims, and
  // value true whenever sparse matrix has a non-zero element over
  // corresponding block and dense dims, and false otherwise.
  auto n_batch_dim = self.dim() - 2 - dense_dim;
  auto is_batched = n_batch_dim > 0;
  auto values = blocked_layout ? _batch_tile_tensor(self, blocksize, dense_dim) :  self;
  auto not_zero_mask = blocked_layout ? _batch_tile_tensor(self_mask, blocksize, dense_dim) : self_mask;
  if (blocked_layout || dense_dim > 0) {
    std::vector<int64_t> reduce_dim((blocked_layout ? 2 : 0) + dense_dim);
    std::iota(reduce_dim.begin(), reduce_dim.end(), n_batch_dim + 2);
    not_zero_mask = not_zero_mask.sum(reduce_dim) != 0;
  }

  if (is_batched) {
    // Prepare for the conversion, in particular join the batch dims
    // and the compressed dim into the single dim.
    dense_to_sparse_compressed_prepare_check_mask_values_batched(
        target_layout, values, not_zero_mask, n_batch_dim);
  }

  // Calculate sparse matrix row and col indices and then, depending
  // on the target layout, corresponding compressed and sparse
  // indices.  Use the mask tensor calculate above to generate sparse
  // matrix values tensor.
  Tensor row_indices;
  Tensor col_indices;
  Tensor compressed_indices;
  if (compressed_rows_layout) {
    std::tie(col_indices, row_indices) = _not_zero_mask_to_col_row_indices(
        not_zero_mask, at::kLong, not_zero_mask.device());
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        row_indices, not_zero_mask.size(0), false /*out_int32*/);
    {
      auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
      values = values.flatten(0, 1).index_select(0, mask_indices);
    }
  } else {
    std::tie(row_indices, col_indices) = _not_zero_mask_to_col_row_indices(
       not_zero_mask.transpose(1, 0), at::kLong, not_zero_mask.device());
    compressed_indices = at::_convert_indices_from_coo_to_csr(
        col_indices, not_zero_mask.size(-1), false /*out_int32*/);
    {
      auto mask_indices = _mask_to_indices(not_zero_mask.transpose(0, 1).flatten());
      values = values.transpose(0, 1).flatten(0, 1).index_select(0, mask_indices);
    }
  }
  Tensor& plain_indices = compressed_rows_layout ? col_indices : row_indices;

  if (is_batched) {
   // Restore the batch dims and compressed dim.
    reshape_2d_sparse_compressed_members_to_nd_batched(
        self.sizes(), n_batch_dim, compressed_indices, plain_indices, values);
  }

  // Create compressed sparse matrix with the target layout.
  return at::_sparse_compressed_tensor_unsafe(
        compressed_indices,
        plain_indices,
        values,
        self.sizes(),
        self.options().layout(target_layout));
}

```

其中关键的两句

``` c++
    {
      auto mask_indices = _mask_to_indices(not_zero_mask.flatten());
      values = values.flatten(0, 1).index_select(0, mask_indices);
    }

```

即，将 `mask` 转为 Tensor 可以 `index_select` 的格式，然后提取出具体值。

# 四、对比分析

Paddle 目前在 `paddle/phi/kernels/sparse/cpu/mask_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/mask_kernel.cu` 中实现了 `MaskCooKernel` 算子：

``` c++
void MaskCooKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const SparseCooTensor& mask,
                   SparseCooTensor* out)
```

此算子实现了与 PyTorch 中 `sparse_mask` 相同的能力，但是，Paddle 算子中只有 `coo` 类型的支持，还需要补充 `csr` 类型稀疏矩阵的支持。

另外，Paddle 的 `MaskCooKernel` 并没有暴露至上层 Python 接口，还需要注册此算子，以及对应的反向算子。

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
// 分为 2D 和 3D 两种处理方式
// 2D
  int64_t numel = 0;
  for (int64_t i = 0; i < mask_crows.numel() - 1; ++i) {
    for (int64_t j = mask_crows.data<IntT>()[i];
         j < mask_crows.data<IntT>()[i + 1];
         ++j) {
      IntT col_idx = mask_cols.data<IntT>()[numel];

      out_values.data<T>()[numel] =
          x.data<T>()[(i / x.dims()[0]) * x.dims()[1] +
                      (i % x.dims()[0]) * x.dims()[1] + col_idx];

      ++numel;
    }
  }

// 3D

  int64_t numel = 0;
  for (int64_t i = 0; i < mask_crows.numel() - 1; ++i) {
    for (int64_t j = mask_crows.data<IntT>()[i];
         j < mask_crows.data<IntT>()[i + 1];
         ++j) {
      IntT col_idx = mask_cols.data<IntT>()[numel];

      out_values.data<T>()[numel] =
          x.data<T>()[(i / x.dims()[0]) * (x.dims()[1] * x.dims()[2]) +
                      (i % x.dims()[0]) * x.dims()[2] + col_idx];

      ++numel;
    }
  }
}
```

这里没有使用类似 PyTorch 的 `index_select` 的转换方式，因为，虽然可以通过 Paddle 的 `masked_select` 实现类似能力，但是，仍需将 `mask` 转换为 `DenseTensor` ，而目前 Paddle 的 `CsrToDenseKernel` 仍然是先将 `csr` 转为 `coo` 然后处理，不具备性能优势。可参考如下代码：

``` c++
template <typename T, typename Context>
void CsrToDenseKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      DenseTensor* out) {
  DenseTensor indices;
  DenseTensor values;
  SparseCooTensor coo(indices, values, x.dims());
  MetaTensor meta_out(&coo);
  phi::UnchangedInferMeta(x, &meta_out);
  CsrToCooKernel<T, Context>(dev_ctx, x, &coo);
  CooToDenseKernel<T, Context>(dev_ctx, coo, out);
}

```

`paddle/phi/kernels/sparse/gpu/mask_kernel.cu` 中 GPU 算子：

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
// 分为 2D 和 3D 两种处理方式
...
}
```

gpu kernel 的实现与 cpu 有所不同，因为 `csr` 格式没有办法直接使用 cuda 的 grid ，需要先把 `csr` 格式的 `crows` 转换为常规 `rows` 的方式，然后复用 gpu kernel 中的 `MaskKernel` 进行实现。

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

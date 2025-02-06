# paddle.baddbmm 设计文档

| API名称                                                            | paddle.baddbmm |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | Qin-sx                                                                                                                       |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2025-01-16                                                                                                                   |
| 版本号                                                             | V1.0                                                                                                                         |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本                                                                                                                  |
| 文件名                                                             | 20250116_api_design_for_baddbmm.md<br>                           |

# 一、概述

## 1、相关背景

https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/%E3%80%90Hackathon_8th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no2-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-baddbmm-api

## 2、功能目标

在 Paddle 框架中新增 baddbmm API：

1. baddbmm API 用于计算 β∗input+α(A@B)。
2. 该 API 支持在 CPU 和 GPU 上运行。
3. 调用路径为：paddle.baddbmm 和 Tensor.baddbmm。

## 3、意义

新增 paddle.baddbmm 丰富 paddle API。

# 二、飞桨现状

飞桨（PaddlePaddle）目前没有提供 baddbmm 的API，需要借助多个函数的组合来实现该功能，这种方式可能会面临精度偏差以及计算效率降低等问题。

# 三、业内方案调研

## PyTorch

### `baddbmm`的实现

#### Python 接口

`baddbmm`函数在/PyTorch/aten/src/ATen/native/native_functions.yaml文件中注册。

```yaml
- func: baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  variants: function, method
  structured_delegate: baddbmm.out

- func: baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
  variants: method
  structured_delegate: baddbmm.out

- func: baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  variants: function
  dispatch:
    CPU: baddbmm_out_cpu
    CUDA: baddbmm_out_cuda
    MPS: baddbmm_out_mps
    XPU: baddbmm_out_xpu
    SparseCsrCUDA: baddbmm_out_sparse_csr_cuda
```

#### C++ 实现

`baddbmm`函数主要由/PyTorch/aten/src/ATen/native/cuda/Blas.cpp文件中的`baddbmm_out_cuda_impl`函数实现。

```C++
TORCH_IMPL_FUNC(baddbmm_out_cuda)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  {
    at::NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, self, batch1, batch2, beta, alpha);
  }
}

const Tensor& baddbmm_out_cuda_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = resolve_conj_if_indicated(result, true);
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ = c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(at::MemoryFormat::Contiguous).transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda = 0, ldb = 0, ldc = 0;
  bool transpose_batch1 = false, transpose_batch2 = false;
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t alpha_val = alpha.to<opmath_t>();
    opmath_t beta_val = beta.to<opmath_t>();
    const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
    const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
    const auto transa = transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n';
    const auto transb = transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n';
    // If batch is 1 call gemm rather than bgemm
    if (num_batches == 1) {
      at::cuda::blas::gemm<scalar_t>(
          transa, transb,
          m, n, k,
          alpha_val,
          batch1_ptr, lda,
          batch2_ptr, ldb,
          beta_val,
          result_ptr, ldc);
    } else {
      at::cuda::blas::bgemm<scalar_t>(
        transa, transb,
        m, n, k,
        alpha_val,
        batch1_ptr, lda, batch1_->strides()[0],
        batch2_ptr, ldb, batch2_->strides()[0],
        beta_val,
        result_ptr, ldc, result_->strides()[0],
        num_batches
      );
   }
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}
```

`baddbmm_out_cuda_impl`函数主要通过文件/PyTorch/aten/src/ATen/cuda/CUDABlas.cpp中`gemm`和`bgemm`函数实现。
`gemm`和`bgemm`函数主要通过调用cublas库的函数来实现批量矩阵乘法。

```C++
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
  else {
    gemm_internal<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#ifdef USE_ROCM
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#endif
  else {
    gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);
#ifdef USE_ROCM
  int flag = 0;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_ex(
      (rocblas_handle)handle,
      hipOperationToRocOperation(opa),
      hipOperationToRocOperation(opb),
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_f16_r,
      ldc,
      c,
      rocblas_datatype_f16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5) {
#ifndef USE_ROCM
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    if (!at::globalContext().allowFP16ReductionCuBLAS()) {
      cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
#endif
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc));
  }
#endif
}

template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    bgemm_internal_cublaslt<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;
#ifdef USE_ROCM
  int flag = 0;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                   hipOperationToRocOperation(opa),
                                   hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                   (void*)&falpha, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                   (void*)&fbeta, c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 5){
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, m, n, k,
      (void*)(&falpha), a, CUDA_R_16F, lda, stridea,
      b, CUDA_R_16F, ldb, strideb, (void*)(&fbeta),
      c, CUDA_R_16F, ldc, stridec,
      num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    for (const auto i : c10::irange(num_batches)) {
      at::cuda::blas::gemm<at::Half>(
        transa, transb,
        m, n, k,
        alpha, (a + i * stridea), lda,
        (b + i * strideb), ldb, beta,
        (c + i * stridec), ldc);
    }
  }
#endif // USE_ROCM
}
```

# 四、对比分析

由于在测试中是将`paddle.baddbmm`函数与`torch.baddbmm`函数进行对比，所以采用类似PyTorch中直接调用cublas库的函数的方式。

# 五、设计思路与实现方案

## 命名与参数设计

API `paddle.baddbmm(input, x, y, alpha=1.0, beta=1.0, name=None)`
paddle.baddbmm
----------------------
参数
- input (Tensor) - 输入 Tensor input，数据类型支持 bfloat16、float16、float32、float64。
- x (Tensor) - 输入 Tensor x，数据类型支持 bfloat16、float16、float32、float64。
- y (Tensor) - 输入 Tensor y，数据类型支持 bfloat16、float16、float32、float64。
- alpha (float，可选) - 乘以 x*y 的标量，数据类型支持 float，默认值为 1.0。
- beta (float，可选) - 乘以 input 的标量，数据类型支持 float，默认值为 1.0。
- name (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

## 底层设计

设计`BaddbmmKernel`函数，调用 blas.h 文件中的`GEMM`和`BatchedGEMM`函数。
为避免精度损失，当数据类型为float16和bfloat16时，参数alpha和beta仍然保持float类型。

```C++
using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // special case for MPType
  if constexpr (std::is_same_v<MPType, float>) {
    VLOG(4) << "Function: baddbmm, Type of T: " << typeid(T).name();
    VLOG(4) << "Function: baddbmm, Type of MPType: " << typeid(MPType).name();
    float t_alpha = alpha;
    float t_beta = beta;
    if (x_dims[0] == 1) {
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                x_dims[1],
                y_dims[2],
                x_dims[2],
                t_alpha,
                x.data<T>(),
                y.data<T>(),
                t_beta,
                out->data<T>());
    } else {
      blas.BatchedGEMM(CblasNoTrans,
                       CblasNoTrans,
                       x_dims[1],
                       y_dims[2],
                       x_dims[2],
                       t_alpha,
                       x.data<T>(),
                       y.data<T>(),
                       t_beta,
                       out->data<T>(),
                       x_dims[0],
                       x_dims[1] * x_dims[2],
                       x_dims[2] * y_dims[2]);
    }
  } else {
    T t_alpha = static_cast<T>(alpha);
    T t_beta = static_cast<T>(beta);
    if (x_dims[0] == 1) {
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                x_dims[1],
                y_dims[2],
                x_dims[2],
                t_alpha,
                x.data<T>(),
                y.data<T>(),
                t_beta,
                out->data<T>());
    } else {
      blas.BatchedGEMM(CblasNoTrans,
                       CblasNoTrans,
                       x_dims[1],
                       y_dims[2],
                       x_dims[2],
                       t_alpha,
                       x.data<T>(),
                       y.data<T>(),
                       t_beta,
                       out->data<T>(),
                       x_dims[0],
                       x_dims[1] * x_dims[2],
                       x_dims[2] * y_dims[2]);
      // x_dims[2] == y_dims[1]
    }
  }
```

在 blas.h 文件中的增加新的函数接口，使`GEMM`和`BatchedGEMM`函数可以传入float类型的alpha和beta参数，而无需与矩阵类型保持一致。

```C++
  template <typename T, typename U = T>
  void GEMM(CBLAS_TRANSPOSE transA,
            CBLAS_TRANSPOSE transB,
            int M,
            int N,
            int K,
            U alpha,
            const T* A,
            const T* B,
            U beta,
            T* C) const;

  template <typename T, typename U = T>
  void BatchedGEMM(CBLAS_TRANSPOSE transA,
                   CBLAS_TRANSPOSE transB,
                   int M,
                   int N,
                   int K,
                   U alpha,
                   const T* A,
                   const T* B,
                   U beta,
                   T* C,
                   int batchCount,
                   int64_t strideA,
                   int64_t strideB) const;
```

## API实现方案

在/paddle/phi/ops/yaml/ops.yaml文件中注册`baddbmm`函数。

```yaml
- op : baddbmm
  args : (Tensor input, Tensor x, Tensor y, float beta=1.0, float alpha=1.0)
  output : Tensor(out)
  infer_meta :
    func : BaddbmmInferMeta
  kernel :
    func : baddbmm
    data_type : x
  inplace: (input -> out)
  backward : baddbmm_grad
  interfaces : paddle::dialect::InferSymbolicShapeInterface
```

# 六、测试和验收的考量

## 单元测试

### 功能测试
  - 计算 β∗input + α(A @ B)。
  - 验证计算结果是否正确。

### 参数测试
  - 测试不同类型的参数。
  - 验证计算结果是否符合预期。

## PyTorch对比测试

### 对比测试

输入不同类型的参数（float32, float16, bfloat16），测试`paddle.baddbmm`和`torch.baddbmm`的结果是否一致。

### 测试代码

```python
import paddle
import torch
import numpy as np

norm_factor = 1.5
batch_size = 4
seq_length = 4096
hidden_size = 8192

beta_input = (1.0 / 1.5)

# paddle_dtype = paddle.float32
# torch_dtype = torch.float32

# paddle_dtype = paddle.float16
# torch_dtype = torch.float16

paddle_dtype = paddle.bfloat16
torch_dtype = torch.bfloat16

query_layer = paddle.randn([seq_length, batch_size, hidden_size]).cast(paddle_dtype)
key_layer = paddle.randn([seq_length, batch_size, hidden_size]).cast(paddle_dtype)

matmul_input_buffer = torch.zeros([batch_size, seq_length, seq_length], dtype=torch_dtype).cuda()
query_layer_torch = torch.tensor(query_layer.cast(paddle.float32).numpy(), dtype=torch_dtype).cuda()
key_layer_torch = torch.tensor(key_layer.cast(paddle.float32).numpy(), dtype=torch_dtype).cuda()

matmul_result_pd = paddle.matmul(query_layer.transpose([1, 0, 2]) * (1.0 / norm_factor), key_layer.transpose([1, 2, 0]))

matmul_result_pt = torch.baddbmm(
    matmul_input_buffer,
    query_layer_torch.transpose(0, 1),  # [b * np, sq, hn]
    key_layer_torch.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    beta=beta_input,
    alpha=(1.0 / norm_factor),
)

matmul_result_pd = matmul_result_pd.cast(paddle.float32)
matmul_result_pt = matmul_result_pt.float()

diff_max = np.max(np.abs(matmul_result_pd.numpy() - matmul_result_pt.cpu().numpy()))
print(f"max diff between paddle matmul and torch baddbmm is {diff_max}")

matmul_input_buffer_paddle = paddle.zeros([batch_size, seq_length, seq_length], dtype=paddle_dtype)
query_layer_torch_paddle = paddle.to_tensor(query_layer, dtype=paddle_dtype)
key_layer_torch_paddle = paddle.to_tensor(key_layer, dtype=paddle_dtype)

matmul_result_pd_baddbmm = paddle.baddbmm(
    matmul_input_buffer_paddle,
    query_layer_torch_paddle.transpose([1, 0, 2]),  # [b * np, sq, hn]
    key_layer_torch_paddle.transpose([1, 2, 0]),  # [b * np, hn, sk]
    beta=beta_input,
    alpha=(1.0 / norm_factor),
)

matmul_result_pd_baddbmm = matmul_result_pd_baddbmm.cast(paddle.float32)

diff_max_paddle = np.max(np.abs(matmul_result_pd_baddbmm.numpy() - matmul_result_pt.cpu().numpy()))
print(f"max diff between paddle baddbmm and torch baddbmm is {diff_max_paddle}")

diff_max2 = np.max(np.abs(matmul_result_pd_baddbmm.numpy() - matmul_result_pd.numpy()))
print(f"max diff between paddle baddbmm and paddle matmul is {diff_max2}")
```

# 七、影响面

## 需要进一步讨论的问题

暂无

## 对二次开发用户的影响

采用增量开发的方式，新增加了`GEMM`和`BatchedGEMM`函数的重载，使得的alpha和beta的类型可以和矩阵类型不一致，用户可以自行选择调用哪种函数接口。


# 八、排期规划

2025/01/24前完成函数实现和测试。

# 九、参考资料

精度对齐：https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/align_pytorch_and_paddle.html

# 十、致谢

感谢朱卫国老师和骆涛老师的指导。

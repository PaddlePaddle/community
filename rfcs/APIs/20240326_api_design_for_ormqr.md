# paddle_ormqr 设计文档

| API 名称     | paddle.ormqr                     |
| ------------ | -------------------------------- |
| 提交作者     | Chen-Lun-Hao                     |
| 提交时间     | 2024-03-27                       |
| 版本号       | V2.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20240326_api_design_for_ormqr.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，需要为飞桨扩充 API `paddle.ormqr`

本 API 属于飞桨开源个人贡献赛 API 开发任务[No.28：为 Paddle 新增 ormqr API](https://github.com/PaddlePaddle/Paddle/issues/62905)的任务。

## 2、功能目标

计算一个普通矩阵与 Householder 矩阵的乘积。计算维度为(m, n)的矩阵 C（由 other 给出）和一个矩阵 Q 的乘积， 其中 Q 由 Householder 反射系数 (x, tau) 表示。

预期该 API 支持

- paddle.linalg.ormqr 作为独立的函数调用
- Tensor.ormqr 作为 Tensor 的方法使用

## 3、意义

为飞桨增加普通矩阵与指定矩阵的乘积的计算方式，提升飞桨 API 丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.ormqr(input, tau, other, left=True, transpose=False, *, out=None) → Tensor` 以及对应的 `torch.Tensor.ormqr`

其介绍为：

> Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix. Multiplies a m×n matrix C (given by other) with a matrix Q, where Q is represented using Householder reflectors (input, tau).

> 参数表为：

- `Tensor` input: tensor of shape (_, mn, k) where _ is zero or more batch dimensions and mn equals to m or n depending on the left.
- `Tensor` tau: tensor of shape (_, min(mn, k)) where _ is zero or more batch dimensions.
- `Tensor` other: tensor of shape (_, m, n) where _ is zero or more batch dimensions.
- `bool` left: controls the order of multiplication.
- `bool` transpose: controls whether the matrix Q is conjugate transposed or not.

### 实现

PyTorch 在 2.2 版本给出的 API 中，其默认后端 Inductor 针对 `ormqr`操作进行实现的代码如下，具体代码可以参考[BatchLinearAlgebraKernel.cpp](https://github.com/pytorch/pytorch/blob/99c822c0ba747fad8528ff6b57712abdbdc2c093/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L2710)

```python
template <typename scalar_t>
void apply_ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.ormqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  char side = left ? 'L' : 'R';
  char trans = transpose ? (input.is_complex() ? 'C' : 'T') : 'N';

  auto input_data = input.const_data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  auto input_matrix_stride = matrixStride(input);
  auto other_matrix_stride = matrixStride(other);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = other.size(-2);
  auto n = other.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, left ? m : n);
  auto ldc = std::max<int64_t>(1, m);
  int info = 0;

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((left ? m : n) >= k);

  // Query for the optimal size of the workspace tensor
  int lwork = -1;
  scalar_t wkopt;
  lapackOrmqr<scalar_t>(side, trans, m, n, k, const_cast<scalar_t*>(input_data), lda, const_cast<scalar_t*>(tau_data), other_data, ldc, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, input.options());

  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual result
    lapackOrmqr<scalar_t>(
        side, trans, m, n, k,
        const_cast<scalar_t*>(input_working_ptr), lda,
        const_cast<scalar_t*>(tau_working_ptr),
        other_working_ptr, ldc,
        work.data_ptr<scalar_t>(), lwork, &info);

    // info from lapackOrmqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_ormqr'
void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "ormqr_cpu", [&]{
    apply_ormqr<scalar_t>(input, tau, other, left, transpose);
  });
}
```

## MindSpore

MindSpore 中有 `mindspore.ops.orqmr` 此接口：

- `mindspore.ops.ormqr(input, tau, other, left=True, transpose=False)`

其实现代码：

https://www.mindspore.cn/docs/zh-CN/master/_modules/mindspore/ops/function/math_func.html#ormqr

```python
class Ormqr(Primitive):
    r"""
    Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.
    Multiplies a(m, n) matrix C (given by other) with a matrix Q, where Q is represented using Householder
    reflectors (x, tau), which is the output of geqrf().

    Refer to :func:`mindspore.ops.ormqr` for more details.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        left (bool, optional): controls the order of multiplication. If ``True`` , compute op(Q)*C.
            If ``False`` , compute C*op(Q). Default: ``True`` .
        transpose(bool, optional): controls whether the matrix Q is conjugate transposed or not.Default: ``False`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, mn, k)` where the value of mn depending on `left`,
          When `left` is ``True``, the value of mn is equal to m; otherwise, the value of mn is equal to n.
          and `*` is zero or more batch dimensions.
        - **tau** (Tensor) - Tensor of shape :math:`(*, min(mn, k))` where `*` is zero or more batch dimensions,
          and its type is the same as `x`.
        - **other** (Tensor) - Tensor of shape :math:`(*, m, n)` where `*` is zero or more batch dimensions,
          and its type is the same as `x`.

    Outputs:
        - **y** (Tensor) - the output Tensor, has the same shape and data type as `other`.

    Raises:
        TypeError: If `x` or `tau` or `other` is not Tensor.
        TypeError: If dtype of `x` or `tau` or `other` is not one of: float64, float32, complex64, complex128.
        ValueError: If `x` or `other` is less than 2D.
        ValueError: If rank(x) - rank(tau) != 1.
        ValueError: If tau.shape[:-2] != x.shape[:-2]
        ValueError: If other.shape[:-2] != x.shape[:-2]
        ValueError: If left == True, other.shape[-2] < tau.shape[-1].
        ValueError: If left == True, other.shape[-2] != x.shape[-2].
        ValueError: If left == False, other.shape[-1] < tau.shape[-1].
        ValueError: If left == False, other.shape[-1] != x.shape[-2].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]]), mindspore.float32)
        >>> tau = Tensor(np.array([1.55, 1.94, 3.0]), mindspore.float32)
        >>> other = Tensor(np.array([[-114.6, 10.9, 1.1],
        ...                          [-0.304, 38.07, 69.38],
        ...                          [-0.45, -0.17, 62]]), mindspore.float32)
        >>> net = ops.Ormqr()
        >>> y = net(x, tau, other)
        >>> print(y)
        [[  63.82713   -13.823125 -116.28614 ]
         [ -53.659264  -28.157839  -70.42702 ]
         [ -79.54292    24.00183   -41.34253 ]]
    """

    @prim_attr_register
    def __init__(self, left=True, transpose=False):
        """Initialize Ormqr"""
        self.init_prim_io_names(inputs=['x', 'tau', 'other'], outputs=['y'])
        self.left = validator.check_value_type('left', left, [bool], self.name)
        self.transpose = validator.check_value_type('transpose', transpose, [bool], self.name)
        self.add_prim_attr('left', self.left)
        self.add_prim_attr('transpose', self.transpose)

def _get_cache_prim(cls: Primitive) -> Primitive:
    """
    Wrapper function, get a primitive by it's all args.

    Args:
        cls (Primitive): The Primitive need be wrapped.

    Returns:
        Function, a new function with return a primitive by it's all args.

    Examples:
        >>> # Example1:
        >>> from mindspore.ops._primitive_cache import _get_cache_prim
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> axis = [0, 1]
        >>> p=2
        >>> keep_dims=False
        >>> epsilon=1e-12
        >>> _lp_norm = _get_cache_prim(P.LpNorm)(axis, p, keep_dims, epsilon)
        >>> output = _lp_norm(input_x)
        >>> print(output)
        [ 9.165152 10.954452]
        >>> # Example2:
        >>> from mindspore.ops._primitive_cache import _get_cache_prim
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> axis = [0, 1]
        >>> _lp_norm = _get_cache_prim(P.LpNorm)(axis, 2, keep_dims=False, epsilon=1e-12)
        >>> output = _lp_norm(input_x)
        >>> print(output)
        [ 9.165152 10.954452]
    """

    def _new_prim_for_graph(*args, **kwargs) -> Primitive:
        return cls(*args, **kwargs)

    def _get_cache_prim_for_pynative(*args, **kwargs) -> Primitive:
        """Get a primitive singleton by it's all args."""
        global _PRIM_CACHE
        key = (str(cls),)
        str_args = [str(arg) for arg in args]
        key += tuple(str_args)
        for attr_name in kwargs:
            attr_value = kwargs.get(attr_name)
            key += (attr_name + ":" + str(attr_value),)
        # Note: The key must be a str.
        key = str(key)
        if key not in _PRIM_CACHE:
            prim = Primitive.__new__(cls, *args, **kwargs)
            # Only init once.
            prim.__init__(*args, **kwargs)
            _PRIM_CACHE[key] = prim
        return _PRIM_CACHE.get(key)

    if _is_need_compile(_temp_func): # @jit.cond: True
        return _new_prim_for_graph
    return _get_cache_prim_for_pynative


[文档]def ormqr(input, tau, other, left=True, transpose=False):
    r"""
    Calculates two matrices multiplication of a product of a general matrix with Householder matrices.
    Calculates the product of a matrix C(given by `other`) with dimensions (m, n) and a matrix Q which is represented
    using Householder reflectors (`input`, `tau`). Returns a Tensor.

    Args:
        input (Tensor): Tensor of shape :math:`(*, mn, k)`, when `left` is True, mn equals to m,
            otherwise, mn equals to n. And `*` is zero or more batch dimensions.
        tau (Tensor): Tensor of shape :math:`(*, min(mn, k))` where `*` is zero or more batch dimensions,
            and its type is the same as `input`.
        other (Tensor): Tensor of shape :math:`(*, m, n)` where `*` is zero or more batch dimensions,
            and its type is the same as `input`.
        left (bool, optional): determines the order of multiplication. If True, computes op(Q) \* `other` ,
            otherwise, compute `other` \* op(Q). Default: ``True`` .
        transpose (bool, optional): If True, the matrix Q is conjugate transposed,
            otherwise, not conjugate transposing matrix Q. Default: ``False`` .

    Returns:
        Tensor, with the same type and shape as `other`.

    Raises:
        TypeError: If `input` or `tau` or `other` is not Tensor.
        TypeError: If dtype of `input` or `tau` or `other` is not one of: float64, float32, complex64, complex128.
        ValueError: If the dimension of `input` or `other` is less than 2D.
        ValueError: If rank(`input`) - rank(`tau`) != 1.
        ValueError: If tau.shape[:-2] != input.shape[:-2]
        ValueError: If other.shape[:-2] != input.shape[:-2]
        ValueError: If left == true, other.shape[-2] < tau.shape[-1].
        ValueError: If left == true, other.shape[-2] != input.shape[-2].
        ValueError: If left == false, other.shape[-1] < tau.shape[-1].
        ValueError: If left == false, other.shape[-1] != input.shape[-2].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]]),
        ...                mindspore.float32)
        >>> tau = Tensor(np.array([1.55, 1.94, 3.0]), mindspore.float32)
        >>> other = Tensor(np.array([[-114.6, 10.9, 1.1],
        ...                          [-0.304, 38.07, 69.38],
        ...                          [-0.45, -0.17, 62]]), mindspore.float32)
        >>> output = ops.ormqr(input, tau, other)
        >>> print(output)
        [[  63.82713   -13.823125 -116.28614 ]
         [ -53.659264  -28.157839  -70.42702 ]
         [ -79.54292    24.00183   -41.34253 ]]
    """

    ormqr_ = _get_cache_prim(Ormqr)(left, transpose)
    return ormqr_(input, tau, other)

```

通过调用 C++接口，实现了 ormqr 操作，具体[代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/plugin/device/gpu/kernel/math/ormqr_gpu_kernel.cc)参考如下：

```
void OrmqrGpuKernelMod::RunOrmqr(T *d_a, T *tau, T *d_c, size_t lda, int *dev_info, T *output_y) {
  int lwork = 0;
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnSormqr_bufferSize(handle_, side_, trans_, m_, n_, x_n_, d_a, lda, tau, d_c, m_, &lwork),
      "cusolver query ormqr work size failed.");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
      cusolverDnDormqr_bufferSize(handle_, side_, trans_, m_, n_, x_n_, d_a, lda, tau, d_c, m_, &lwork),
      "cusolver query ormqr work size failed.");
  } else {
    if constexpr (std::is_same_v<T, Complex<float>>) {
      trans_ = transpose_ ? CUBLAS_OP_C : CUBLAS_OP_N;
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnCunmqr_bufferSize(handle_, side_, trans_, m_, n_, x_n_, reinterpret_cast<cuComplex *>(d_a), lda,
                                    reinterpret_cast<cuComplex *>(tau), reinterpret_cast<cuComplex *>(d_c), m_, &lwork),
        "cusolver query ormqr work size failed.");
    }
    if constexpr (std::is_same_v<T, Complex<double>>) {
      trans_ = transpose_ ? CUBLAS_OP_C : CUBLAS_OP_N;
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnZunmqr_bufferSize(handle_, side_, trans_, m_, n_, x_n_, reinterpret_cast<cuDoubleComplex *>(d_a), lda,
                                    reinterpret_cast<cuDoubleComplex *>(tau), reinterpret_cast<cuDoubleComplex *>(d_c),
                                    m_, &lwork),
        "cusolver query ormqr work size failed.");
    }
  }

  void *d_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(sizeof(T) * lwork);
  if (d_work == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the memory of d_work alloc failed.";
  }
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSormqr(handle_, side_, trans_, m_, n_, x_n_, d_a, lda, tau, d_c,
                                                            m_, static_cast<T *>(d_work), lwork, dev_info),
                                           "cusolver ormqr failed.");
  } else if constexpr (std::is_same_v<T, double>) {
    CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDormqr(handle_, side_, trans_, m_, n_, tau_n_, d_a, lda, tau, d_c,
                                                            m_, static_cast<T *>(d_work), lwork, dev_info),
                                           "cusolver ormqr failed.");
  } else {
    if constexpr (std::is_same_v<T, Complex<float>>) {
      trans_ = transpose_ ? CUBLAS_OP_C : CUBLAS_OP_N;
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnCunmqr(handle_, side_, trans_, m_, n_, x_n_, reinterpret_cast<cuComplex *>(d_a), lda,
                         reinterpret_cast<cuComplex *>(tau), reinterpret_cast<cuComplex *>(d_c), m_,
                         reinterpret_cast<cuComplex *>(d_work), lwork, dev_info),
        "cusolver ormqr failed.");
    }
    if constexpr (std::is_same_v<T, Complex<double>>) {
      trans_ = transpose_ ? CUBLAS_OP_C : CUBLAS_OP_N;
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnZunmqr(handle_, side_, trans_, m_, n_, x_n_, reinterpret_cast<cuDoubleComplex *>(d_a), lda,
                         reinterpret_cast<cuDoubleComplex *>(tau), reinterpret_cast<cuDoubleComplex *>(d_c), m_,
                         reinterpret_cast<cuDoubleComplex *>(d_work), lwork, dev_info),
        "cusolver ormqr failed.");
    }
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_y, d_c, sizeof(T) * m_ * n_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cuda memcpy output A failed!");
  device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(d_work);
}

void OrmqrGpuKernelMod::CheckResult(int *dev_info) {
  std::vector<int> info_gpu(batch_size_, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(info_gpu.data(), dev_info, sizeof(int) * batch_size_, cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Copy device result failed");
  for (size_t i = 0; i < info_gpu.size(); ++i) {
    if (info_gpu[i] != 0) {
      MS_LOG(INFO) << "For '" << kernel_name_ << "', the compute result has wrong value. The " << -info_gpu[i]
                   << "th parameter is wrong (not counting handle) in batch " << i << " data.";
    }
  }
}

template <typename T>
void OrmqrGpuKernelMod::LaunchOrmqr(T *d_input_x, T *input_tau, T *d_input_other, T *d_output_y, int *dev_info) {
  size_t lda = m_;
  if (side_ == CUBLAS_SIDE_RIGHT) {
    lda = n_;
  }
  for (size_t batch = 0; batch < batch_size_; ++batch) {
    RunOrmqr(d_input_x + batch * x_m_ * x_n_, input_tau + batch * tau_n_, d_input_other + batch * m_ * n_, lda,
             dev_info + batch, d_output_y + batch * m_ * n_);
  }

  CheckResult(dev_info);
}

```

# 四、对比分析

在 Pytorch 以及 MindSpore 框架中，他们对于 ormqr 算子的实现方式不同，Pytorch 中通过对 LAPAC 的 ormqr 进行封装来提供 ormqr 接口，而 MindSpore 则是自行使用 C++代码设计 ormqr 底层操作。

# 五、设计思路与实现方案

paddle 目前的算子已经支持矩阵的转置,行列计算等操作，因此，可以使用 paddle 已有算子实现 `ormqr` 。

## 命名与参数设计

添加 Python API:

```python
paddle.orqmr(input, tau, other, left=True, transpose=False)
```

参数表：

- input: (Tensor) shape（\*，mn，k），当 left 为 True 时， mn 的值等于 m，否则 mn 的值等于 n。 \*表示 Tensor 在轴 0 上的长度为 0 或者大于 0。
- tau: (Tensor) shape（\*，min（mn，k）），其中 \_ 表示 Tensor 在轴 0 上的长度为 0 或者大于 0，其类型与 input 相同。
- other: (Tensor) shape（\*，m，n），其中 \* 表示 Tensor 在轴 0 上的长度为 0 或者大于 0，其类型与 input 相同。
- left: (bool, 可选) 决定了矩阵乘积运算的顺序。如果 left 为 True ，计算顺序为 op(Q) ∗ other ，否则，计算顺序为 other \* op(Q)。默认值：True。
- transpose: (bool, 可选) 如果为 True ，对矩阵 Q 进行共轭转置变换，否则，不对矩阵 Q 进行共轭转置变换。默认值： False。

## 底层 OP 设计

不涉及底层 OP。

# 六、测试和验收的考量

- 支持 CPU、GPU 测试场景
- 支持动态图以及静态图
- 支持各种 Tensor，如：float32, float64, complex64, complex128
- 通过对比 mindspore 框架中的 ormqr 算子输出，计算结果是否一致
- 需要检查二到三维矩阵的计算情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关 PythonAPI 均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

# 名词解释

# 附件及参考资料

[【Hackathon 6th No.4】为 Paddle 新增 ormqr API](https://github.com/PaddlePaddle/community/pull/668)

[PyTorch ormqr 文档](https://pytorch.org/docs/stable/generated/torch.ormqr.html)

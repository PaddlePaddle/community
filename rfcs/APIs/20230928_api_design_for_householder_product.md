# paddle.linalg.householder_product设计文档

| API 名称     | paddle.linalg.householder_product           |
| ------------ | ------------------------------------------- |
| 提交作者     | coco                                        |
| 提交时间     | 2023-10-16                                  |
| 版本号       | V1.0                                        |
| 依赖飞桨版本 | develop                                     |
| 文件名       | 20230928_api_design_for_householder_product |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，Paddle需要扩充API，调用路径为：

- paddle.linalg.householder_product

## 2、功能目标

计算 Householder 矩阵乘积的前 n 列。

## 3、意义

飞桨支持利用lapack的geqrf返回得到的A和tau，进行计算得到QR分解后的Q矩阵

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch中有API `torch.linalg.householder_product(A, tau, *, out=None) → Tensor` 

在PyTorch中介绍为：

```
Computes the first n columns of a product of Householder matrices.

```

## 实现方法

从实现方法上，PyTorch中`householder_product`与`orgqr`一致，[代码位置](https://github.com/pytorch/pytorch/blob/9af82fa2b86fb71df503082b1960c9392f9dc66d/aten/src/ATen/native/BatchLinearAlgebra.cpp#L2449C2-L2497)

```cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(orgqr_stub);

/*
  The householder_product (orgqr) function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as is produced by the geqrf function.

  Args:
  * `input` - Tensor with the directions of the elementary reflectors below the diagonal.
  * `tau` - Tensor containing the magnitudes of the elementary reflectors.
  * `result` - result Tensor, which will contain the orthogonal (or unitary) matrix Q.

  For further details, please see the LAPACK/MAGMA documentation.
*/
static Tensor& householder_product_out_helper(const Tensor& input, const Tensor& tau, Tensor& result) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-2) >= input.size(-1));
  TORCH_INTERNAL_ASSERT(input.size(-1) >= tau.size(-1));

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // tau tensor must be contiguous
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // orgqr_stub (apply_orgqr) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  result = orgqr_stub(result.device().type(), result, tau_);
  return result;
}
```

kernel实现，[代码位置](https://github.com/pytorch/pytorch/blob/9af82fa2b86fb71df503082b1960c9392f9dc66d/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L445-L451)

```cpp
REGISTER_ARCH_DISPATCH(orgqr_stub, DEFAULT, &orgqr_kernel_impl);

...

// This is a type dispatching helper function for 'apply_orgqr'
Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cpu", [&]{
    apply_orgqr<scalar_t>(result, tau);
  });
  return result;
}
```

然后调用外部lapack包中的`orgqr`,[代码位置](https://github.com/pytorch/pytorch/blob/9af82fa2b86fb71df503082b1960c9392f9dc66d/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L390-L443)

```cpp
/*
  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as produced by the geqrf function.

  Args:
  * `self` - Tensor with the directions of the elementary reflectors below the diagonal,
              it will be overwritten with the result
  * `tau` - Tensor containing the magnitudes of the elementary reflectors

  For further details, please see the LAPACK documentation for ORGQR and UNGQR.
*/
template <typename scalar_t>
inline void apply_orgqr(Tensor& self, const Tensor& tau) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.orgqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // Some LAPACK implementations might not work well with empty matrices:
  // workspace query might return lwork as 0, which is not allowed (requirement is lwork >= 1)
  // We don't need to do any calculations in this case, so let's return early
  if (self.numel() == 0) {
    return;
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(self);
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, m);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackOrgqr<scalar_t>(m, n, k, self_data, lda, tau_data, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual Q
    lapackOrgqr<scalar_t>(m, n, k, self_working_ptr, lda, tau_working_ptr, work.data_ptr<scalar_t>(), lwork, &info);

    // info from lapackOrgqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}
```

lapack调用相关，[代码位置1](https://github.com/pytorch/pytorch/blob/9af82fa2b86fb71df503082b1960c9392f9dc66d/aten/src/ATen/native/BatchLinearAlgebra.cpp#L876-L878)，[2](https://github.com/pytorch/pytorch/blob/9af82fa2b86fb71df503082b1960c9392f9dc66d/aten/src/ATen/native/BatchLinearAlgebra.cpp#L279)

```cpp
template<> void lapackOrgqr<c10::complex<float>>(int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cungqr_(&m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

...
    
extern "C" void cungqr_(int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
```



## Scipy

在`scipy.linalg.lapack`中有与`householder_product`等价的`orgqr`接口，分别支持`float`,`double`,`std::complex<double>`,`std::complex<float>`四种数据类型

| [`sorgqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.sorgqr.html#scipy.linalg.lapack.sorgqr)(a,tau,[lwork,overwrite_a]) | Wrapper for `sorgqr`. |
| ------------------------------------------------------------ | --------------------- |
| [`dorgqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dorgqr.html#scipy.linalg.lapack.dorgqr)(a,tau,[lwork,overwrite_a]) | Wrapper for `dorgqr`. |
| [`zungqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.zungqr.html#scipy.linalg.lapack.zungqr)(a,tau,[lwork,overwrite_a]) | Wrapper for `zungqr`. |
| [`cungrq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.cungrq.html#scipy.linalg.lapack.cungrq)(a,tau,[lwork,overwrite_a]) | Wrapper for `cungrq`. |



## TensorFlow

无`houholder_product`实现，不过有相关的`qr`分解计算，[代码位置](https://github.com/tensorflow/tensorflow/blob/715f951eb9ca20fdcef20bb544b74dbe576734da/tensorflow/core/kernels/qr_op_impl.h#L83-L102)

```cpp
void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                   MatrixMaps* outputs) final {
    Eigen::HouseholderQR<Matrix> qr(inputs[0]);
    const int m = inputs[0].rows();
    const int n = inputs[0].cols();
    const int min_size = std::min(m, n);

    if (full_matrices_) {
        outputs->at(0) = qr.householderQ();
        outputs->at(1) = qr.matrixQR().template triangularView<Eigen::Upper>();
    } else {
        // TODO(jpoulson): Exploit the fact that Householder transformations can
        // be expanded faster than they can be applied to an arbitrary matrix
        // (Cf. LAPACK's DORGQR).
        Matrix tmp = Matrix::Identity(m, min_size);
        outputs->at(0) = qr.householderQ() * tmp;
        auto qr_top = qr.matrixQR().block(0, 0, min_size, n);
        outputs->at(1) = qr_top.template triangularView<Eigen::Upper>();
    }
}
```

可见tf中是用`eigen`库实现`qr`分解的，但是这里是直接得到`Q`，而非`householder_product`要实现的功能。







# 四、对比分析

PyTorch通过底层调用lapack包来实现householder_product，Scipy中也是同样通过lapack包实现相关功能，tf中则是利用eigen库来做qr分解，并没有实现householder_product的功能。

# 五、设计思路与实现方案

在 Paddle repo 的 python/paddle/linalg.py 文件中直接实现

## 命名与参数设计

API的设计为:

- torch.linalg.householder_product(A, tau, name=None)

其中

+ A(Tensor)：shape 为 (*,m,n),至少为2维。
+ tau(Tensor)：shape 为 (*,k)，至少1维。
+ name(str)：表示算子名称，与其他算子统一，默认为None。

返回 out(Tensor): `out=H[0]*H[1]*H[2]...H[n-1]`,即QR分解中的Q

## 底层OP设计

目前暂时直接在python层实现（实现与lapack中相同的算法），以下为同样可行的方案(具体可见[issue中的讨论](https://github.com/PaddlePaddle/community/pull/703#discussion_r1369615670)):

> 在cpp层实现，直接调用lapack，此方法需要再引入一些外部lapack函数，并封装到`lapack_function`中（因为paddle中还没引入`orgqr`等lapack包，相关的qr分解并没有使用lapack而是调了eigen和cusolve来做），最后写cpp kernel调用对应的`lapack_function`实现，再封装为python API。



## API实现方案

geqrf处理过A之后，得到压缩的A和tau，即此时A上三角部分表示QR分解中的R，在上三角之外的下部分存放了`w`信息。将tau与压缩在A中的`w`一起得到`H`，然后根据`Q=H[0]*H[1]*H[2]...H[n-1]`得到`Q`

# 六、测试和验收的考量

单测代码路径：Paddle repo 的 test/test_householder_product.py

测试计算householder矩阵，并结合QR分解得到的Q进行验证

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.linalg.householder_product.html#torch.linalg.householder_product)


# paddle_linalg_matrix_exp 设计文档

| API名称                                                      | paddle.linalg.matrix_exp                  |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | megemini (柳顺)                                    |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-12-06                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   |
| 文件名                                                       | 20231206_api_design_for_matrix_exp.md<br> |


# 一、概述
## 1、相关背景
`matrix_exp` 用于计算矩阵的指数函数 (matrix exponential)，常用于解决一阶线性常微分方程组、马尔可夫链的演化等数学问题，新增此 API 适用于解决数学建模、物理建模和科学计算的相关问题。

## 2、功能目标

对应 PyTorch 的 `torch.linalg.matrix_exp` 操作，调用路径为`paddle.linalg.matrix_exp`，用于计算方阵 $$x$$ 的指数函数，即 $$e^x$$ 。对于 $$x \in \mathbb{K}^{n \times n}$$ ，其中 $$\mathbb{K}$$ 可以是实数集 $$\mathbb{R}$$ 或复数集 $$\mathbb{C}$$，其定义如下：即使用矩阵指数的经典定义——泰勒级数，采用幂级数的方式表示，定义如下：

$$
\mathrm{matrix\_exp}(x) = \sum_{k=0}^\infty \frac{1}{k!}x^k \in \mathbb{K}^{n \times n}
$$

如果矩阵 $$A$$ 具有特征值 $$\lambda_i \in \mathbb{C}$$，那么矩阵 $$\mathrm{matrix\_exp}(A)$$ 的特征值为 $$e^{\lambda_i} \in \mathbb{C}$$。

API 需要输入一个参数 `input`，对应上式的 $$x$$

## 3、意义
为 Paddle 增加用于计算矩阵的指数函数的 API 功能，丰富`paddle`支持线性代数运算的 API。

# 二、飞桨现状
目前 Paddle 在矩阵的函数运算功能中缺少 `paddle.linalg.matrix_exp` API，无法方便地计算矩阵进行指数运算的结果。


# 三、业内方案调研
## PyTorch

pyTorch 中对应的 API 函数为：`torch.linalg.matrix_exp`

在PyTorch中，介绍为：

```
Computes the matrix exponential of a square matrix.
```

实现的代码如下：
```cpp
// matrix exponential
Tensor mexp(const Tensor& a, bool compute_highest_degree_approx = false) {
  // squash batch dimensions to one dimension for simplicity
  const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});

  if (a.scalar_type() == at::ScalarType::Float
      || a.scalar_type() == at::ScalarType::ComplexFloat) {
    constexpr std::array<float, total_n_degs> thetas_float = {
      1.192092800768788e-07, // deg 1
      5.978858893805233e-04, // deg 2
      5.116619363445086e-02, // deg 4
      5.800524627688768e-01, // deg 8
      1.461661507209034e+00, // deg 12
      3.010066362817634e+00  // deg 18
    };

    return mexp_impl<float>(a_3d, thetas_float, compute_highest_degree_approx)
      .view(a.sizes());
  }
  else { // if Double or ComplexDouble
    constexpr std::array<double, total_n_degs> thetas_double = {
      2.220446049250313e-16, // deg 1
      2.580956802971767e-08, // deg 2
      3.397168839976962e-04, // deg 4
      4.991228871115323e-02, // deg 8
      2.996158913811580e-01, // deg 12
      1.090863719290036e+00  // deg 18
    };

    return mexp_impl<double>(a_3d, thetas_double, compute_highest_degree_approx)
      .view(a.sizes());
  }
}

```
在CPU和CUDA上都是基于mexp_impl实现:
```cpp
template <typename scalar_t>
Tensor mexp_impl(
  const Tensor& a,
  std::array<scalar_t, total_n_degs> thetas,
  bool compute_highest_degree_approx = false
)
```
其中核心计算逻辑是基于compute_T18_scale_square　[(参考论文　Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation)](https://www.mdpi.com/2227-7390/7/12/1174)　进行了scale和square近似计算矩阵指数，减少了很多乘积操作，其中大量运用了torch中的基本算子:
```cpp
template <typename scalar_t>
void compute_T18_scale_square(
  Tensor& mexp_out,
  const Tensor& a,
  const Tensor& norm,
  scalar_t theta
) {
  // Scale
  const auto s = at::max(
    at::zeros_like(norm),
    at::ceil(at::log2(norm / theta))
  ).unsqueeze(-1).unsqueeze(-1).to(at::kLong);
  const auto pow2s = at::pow(2, s);
  const auto a_scaled = a / pow2s;

  // Square
  auto mexp_scaled = at::native::compute_T18<scalar_t>(a_scaled);
  auto s_cpu = (s.device().type() == at::kCPU)
    ? s : s.to(at::kCPU);
  for (const auto i : c10::irange(mexp_scaled.size(0))) {
    auto s_val = s_cpu.select(0, i).template item<int64_t>();
    auto mexp = mexp_scaled.select(0, i);
    for (const auto p : c10::irange(s_val)) {
      (void)p; //Suppress unused variable warning
      mexp = at::matmul(mexp, mexp);
    }
    mexp_out.select(0, i).copy_(mexp);
  }
}
```
反向传播计算逻辑如下，其中`function_of_a_matrix`即为`matrix_exp`:
```cpp
template <typename func_t>
Tensor backward_analytic_function_of_a_matrix(
    const Tensor& self, const Tensor& grad,
    const func_t& function_of_a_matrix
  ) {
  auto self_transposed = self.mH();
  auto self_transposed_sizes = self_transposed.sizes().vec();
  self_transposed_sizes[self.dim() - 2] <<= 1;
  self_transposed_sizes[self.dim() - 1] <<= 1;

  auto n = self_transposed.size(-1);
  auto meta_grad = at::zeros(self_transposed_sizes, grad.options());
  meta_grad.narrow(-2, 0, n).narrow(-1, 0, n).copy_(self_transposed);
  meta_grad.narrow(-2, n, n).narrow(-1, n, n).copy_(self_transposed);
  meta_grad.narrow(-2, 0, n).narrow(-1, n, n).copy_(grad);

  auto grad_input = function_of_a_matrix(meta_grad)
    .narrow(-2, 0, n).narrow(-1, n, n);
  return grad_input;
}
```
最后， `torch.linalg.matrix_exp` 支持以下数据类型：bfloat16、float、double、cfloat 和 cdouble。

## TensorFlow
TensorFlow 中对应的 API 函数为：`tf.linalg.expm`，通过 Python API 的方式实现。
[代码位置](https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/linalg/linalg_impl.py#L232-L347)
```py
@tf_export('linalg.expm')
@dispatch.add_dispatch_support
def matrix_exponential(input, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the matrix exponential of one or more square matrices.
  $$exp(A) = \sum_{n=0}^\infty A^n/n!$$
  The exponential is computed using a combination of the scaling and squaring
  method and the Pade approximation. Details can be found in:
  Nicholas J. Higham, "The scaling and squaring method for the matrix
  exponential revisited," SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005.
  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the exponential for all input submatrices `[..., :, :]`.
  Args:
    input: A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`, or
      `complex128` with shape `[..., M, M]`.
    name:  A name to give this `Op` (optional).
  Returns:
    the matrix exponential of the input.
  Raises:
    ValueError: An unsupported type is provided as input.
  @compatibility(scipy)
  Equivalent to scipy.linalg.expm
  @end_compatibility
  """
  with ops.name_scope(name, 'matrix_exponential', [input]):
    matrix = ops.convert_to_tensor(input, name='input')
    if matrix.shape[-2:] == [0, 0]:
      return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
      batch_shape = array_ops.shape(matrix)[:-2]
    # reshaping the batch makes the where statements work better
    matrix = array_ops.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    l1_norm = math_ops.reduce_max(
        math_ops.reduce_sum(
            math_ops.abs(matrix),
            axis=array_ops.size(array_ops.shape(matrix)) - 2),
        axis=-1)[..., array_ops.newaxis, array_ops.newaxis]
    const = lambda x: constant_op.constant(x, l1_norm.dtype)
    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return array_ops.where_v2(
            math_ops.less(l1_norm, const(vals[0])), cases[0], cases[1])
      else:
        return array_ops.where_v2(
            math_ops.less(l1_norm, const(vals[0])), cases[0],
            _nest_where(vals[1:], cases[1:]))
    if matrix.dtype in [dtypes.float16, dtypes.float32, dtypes.complex64]:
      maxnorm = const(3.925724783138660)
      squarings = math_ops.maximum(
          math_ops.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix)
      u5, v5 = _matrix_exp_pade5(matrix)
      u7, v7 = _matrix_exp_pade7(
          matrix /
          math_ops.cast(math_ops.pow(const(2.0), squarings), matrix.dtype))
      conds = (4.258730016922831e-001, 1.880152677804762e+000)
      u = _nest_where(conds, (u3, u5, u7))
      v = _nest_where(conds, (v3, v5, v7))
    elif matrix.dtype in [dtypes.float64, dtypes.complex128]:
      maxnorm = const(5.371920351148152)
      squarings = math_ops.maximum(
          math_ops.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix)
      u5, v5 = _matrix_exp_pade5(matrix)
      u7, v7 = _matrix_exp_pade7(matrix)
      u9, v9 = _matrix_exp_pade9(matrix)
      u13, v13 = _matrix_exp_pade13(
          matrix /
          math_ops.cast(math_ops.pow(const(2.0), squarings), matrix.dtype))
      conds = (1.495585217958292e-002, 2.539398330063230e-001,
               9.504178996162932e-001, 2.097847961257068e+000)
      u = _nest_where(conds, (u3, u5, u7, u9, u13))
      v = _nest_where(conds, (v3, v5, v7, v9, v13))
    else:
      raise ValueError('tf.linalg.expm does not support matrices of type %s' %
                       matrix.dtype)
    is_finite = math_ops.is_finite(math_ops.reduce_max(l1_norm))
    nan = constant_op.constant(np.nan, matrix.dtype)
    result = tf_cond.cond(
        is_finite, lambda: linalg_ops.matrix_solve(-u + v, u + v),
        lambda: array_ops.fill(array_ops.shape(matrix), nan))
    max_squarings = math_ops.reduce_max(squarings)
    i = const(0.0)
    def c(i, _):
      return tf_cond.cond(is_finite,
                          lambda: math_ops.less(i, max_squarings),
                          lambda: constant_op.constant(False))
    def b(i, r):
      return i + 1, array_ops.where_v2(
          math_ops.less(i, squarings), math_ops.matmul(r, r), r)
    _, result = while_loop.while_loop(c, b, [i, result])
    if not matrix.shape.is_fully_defined():
      return array_ops.reshape(
          result,
          array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
    return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))
```

## Scipy

在 Scipy 中使用的 API 格式如下：
`scipy.linalg.expm(A)`[(参考API文档)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)
其中，`Ａ`表示要计算的矩阵，其中最后二维要求是方阵。
Scipy的计算逻辑主要参考论文 [A new scaling and squaring algorithm for the matrix exponential](https://www.researchgate.net/publication/43194029_A_New_Scaling_and_Squaring_Algorithm_for_the_Matrix_Exponential) 以及　
[A Block Algorithm　for Matrix 1-Norm Estimation, with an Application to 1-Norm　Pseudospectra](https://eprints.maths.manchester.ac.uk/321/1/35608.pdf) 。
[实现的代码片段](https://github.com/scipy/scipy/blob/main/scipy/linalg/_matfuncs.py#L214) 如下：
```python
def expm(A):
    a = np.asarray(A)
    if a.size == 1 and a.ndim < 2:
        return np.array([[np.exp(a.item())]])
    if a.ndim < 2:
        raise LinAlgError('The input array must be at least two-dimensional')
    if a.shape[-1] != a.shape[-2]:
        raise LinAlgError('Last 2 dimensions of the array must be square')
    n = a.shape[-1]
    # Empty array
    if min(*a.shape) == 0:
        return np.empty_like(a)
    # Scalar case
    if a.shape[-2:] == (1, 1):
        return np.exp(a)
    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(float)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)
    # Explicit formula for 2x2 case, formula (2.2) in [1]
    # without Kahan's method numerical instabilities can occur.
    if a.shape[-2:] == (2, 2):
        a1, a2, a3, a4 = (a[..., [0], [0]],
                          a[..., [0], [1]],
                          a[..., [1], [0]],
                          a[..., [1], [1]])
        mu = csqrt((a1-a4)**2 + 4*a2*a3)/2.  # csqrt slow but handles neg.vals
        eApD2 = np.exp((a1+a4)/2.)
        AmD2 = (a1 - a4)/2.
        coshMu = np.cosh(mu)
        sinchMu = np.ones_like(coshMu)
        mask = mu != 0
        sinchMu[mask] = np.sinh(mu[mask]) / mu[mask]
        eA = np.empty((a.shape), dtype=mu.dtype)
        eA[..., [0], [0]] = eApD2 * (coshMu + AmD2*sinchMu)
        eA[..., [0], [1]] = eApD2 * a2 * sinchMu
        eA[..., [1], [0]] = eApD2 * a3 * sinchMu
        eA[..., [1], [1]] = eApD2 * (coshMu - AmD2*sinchMu)
        if np.isrealobj(a):
            return eA.real
        return eA
    # larger problem with unspecified stacked dimensions.
    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # working memory to hold intermediate arrays
    Am = np.empty((5, n, n), dtype=a.dtype)
    # Main loop to go through the slices of an ndarray and passing to expm
    for ind in product(*[range(x) for x in a.shape[:-2]]):
        aw = a[ind]
        lu = bandwidth(aw)
        if not any(lu):  # a is diagonal?
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue
        # Generic/triangular case; copy the slice into scratch and send.
        # Am will be mutated by pick_pade_structure
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)
        if s != 0:  # scaling needed
            Am[:4] *= [[[2**(-s)]], [[4**(-s)]], [[16**(-s)]], [[64**(-s)]]]
        pade_UV_calc(Am, n, m)
        eAw = Am[0]
        if s != 0:  # squaring needed
            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # This branch implements Code Fragment 2.1 of [1]
                diag_aw = np.diag(aw)
                # einsum returns a writable view
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # super/sub diagonal
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)
                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw
                    # diagonal
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd
            else:  # generic
                for _ in range(s):
                    eAw = eAw @ eAw
        # Zero out the entries from np.empty in case of triangular input
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw
    return eA
```

## Eigen

Eigen-unsupported 中实现了 matrix exp 接口：

- https://eigen.tuxfamily.org/dox/unsupported/MatrixExponential_8h_source.html

``` cpp
 // This file is part of Eigen, a lightweight C++ template library
 // for linear algebra.
 //
 // Copyright (C) 2009, 2010, 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
 // Copyright (C) 2011, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
 //
 // This Source Code Form is subject to the terms of the Mozilla
 // Public License v. 2.0. If a copy of the MPL was not distributed
 // with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

 #ifndef EIGEN_MATRIX_EXPONENTIAL
 #define EIGEN_MATRIX_EXPONENTIAL

 #include "StemFunction.h"

 #include "./InternalHeaderCheck.h"

 namespace Eigen {
 namespace internal {

 template <typename RealScalar>
 struct MatrixExponentialScalingOp
 {
   MatrixExponentialScalingOp(int squarings) : m_squarings(squarings) { }


   inline const RealScalar operator() (const RealScalar& x) const
   {
     using std::ldexp;
     return ldexp(x, -m_squarings);
   }

   typedef std::complex<RealScalar> ComplexScalar;

   inline const ComplexScalar operator() (const ComplexScalar& x) const
   {
     using std::ldexp;
     return ComplexScalar(ldexp(x.real(), -m_squarings), ldexp(x.imag(), -m_squarings));
   }

   private:
     int m_squarings;
 };

 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade3(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatA>::Scalar>::Real RealScalar;
   const RealScalar b[] = {120.L, 60.L, 12.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType tmp = b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   V = b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
 }

 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade5(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   const RealScalar b[] = {30240.L, 15120.L, 3360.L, 420.L, 30.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType A4 = A2 * A2;
   const MatrixType tmp = b[5] * A4 + b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   V = b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
 }

 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade7(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   const RealScalar b[] = {17297280.L, 8648640.L, 1995840.L, 277200.L, 25200.L, 1512.L, 56.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType A4 = A2 * A2;
   const MatrixType A6 = A4 * A2;
   const MatrixType tmp = b[7] * A6 + b[5] * A4 + b[3] * A2
     + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());

 }

 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade9(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   const RealScalar b[] = {17643225600.L, 8821612800.L, 2075673600.L, 302702400.L, 30270240.L,
                           2162160.L, 110880.L, 3960.L, 90.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType A4 = A2 * A2;
   const MatrixType A6 = A4 * A2;
   const MatrixType A8 = A6 * A2;
   const MatrixType tmp = b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2
     + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
 }

 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade13(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   const RealScalar b[] = {64764752532480000.L, 32382376266240000.L, 7771770303897600.L,
                           1187353796428800.L, 129060195264000.L, 10559470521600.L, 670442572800.L,
                           33522128640.L, 1323241920.L, 40840800.L, 960960.L, 16380.L, 182.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType A4 = A2 * A2;
   const MatrixType A6 = A4 * A2;
   V = b[13] * A6 + b[11] * A4 + b[9] * A2; // used for temporary storage
   MatrixType tmp = A6 * V;
   tmp += b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   tmp = b[12] * A6 + b[10] * A4 + b[8] * A2;
   V.noalias() = A6 * tmp;
   V += b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * MatrixType::Identity(A.rows(), A.cols());
 }

 #if LDBL_MANT_DIG > 64
 template <typename MatA, typename MatU, typename MatV>
 void matrix_exp_pade17(const MatA& A, MatU& U, MatV& V)
 {
   typedef typename MatA::PlainObject MatrixType;
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   const RealScalar b[] = {830034394580628357120000.L, 415017197290314178560000.L,
                           100610229646136770560000.L, 15720348382208870400000.L,
                           1774878043152614400000.L, 153822763739893248000.L, 10608466464820224000.L,
                           595373117923584000.L, 27563570274240000.L, 1060137318240000.L,
                           33924394183680.L, 899510451840.L, 19554575040.L, 341863200.L, 4651200.L,
                           46512.L, 306.L, 1.L};
   const MatrixType A2 = A * A;
   const MatrixType A4 = A2 * A2;
   const MatrixType A6 = A4 * A2;
   const MatrixType A8 = A4 * A4;
   V = b[17] * A8 + b[15] * A6 + b[13] * A4 + b[11] * A2; // used for temporary storage
   MatrixType tmp = A8 * V;
   tmp += b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2
     + b[1] * MatrixType::Identity(A.rows(), A.cols());
   U.noalias() = A * tmp;
   tmp = b[16] * A8 + b[14] * A6 + b[12] * A4 + b[10] * A2;
   V.noalias() = tmp * A8;
   V += b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2
     + b[0] * MatrixType::Identity(A.rows(), A.cols());
 }
 #endif

 template <typename MatrixType, typename RealScalar = typename NumTraits<typename traits<MatrixType>::Scalar>::Real>
 struct matrix_exp_computeUV
 {
   static void run(const MatrixType& arg, MatrixType& U, MatrixType& V, int& squarings);
 };

 template <typename MatrixType>
 struct matrix_exp_computeUV<MatrixType, float>
 {
   template <typename ArgType>
   static void run(const ArgType& arg, MatrixType& U, MatrixType& V, int& squarings)
   {
     using std::frexp;
     using std::pow;
     const float l1norm = arg.cwiseAbs().colwise().sum().maxCoeff();
     squarings = 0;
     if (l1norm < 4.258730016922831e-001f) {
       matrix_exp_pade3(arg, U, V);
     } else if (l1norm < 1.880152677804762e+000f) {
       matrix_exp_pade5(arg, U, V);
     } else {
       const float maxnorm = 3.925724783138660f;
       frexp(l1norm / maxnorm, &squarings);
       if (squarings < 0) squarings = 0;
       MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<float>(squarings));
       matrix_exp_pade7(A, U, V);
     }
   }
 };

 template <typename MatrixType>
 struct matrix_exp_computeUV<MatrixType, double>
 {
   typedef typename NumTraits<typename traits<MatrixType>::Scalar>::Real RealScalar;
   template <typename ArgType>
   static void run(const ArgType& arg, MatrixType& U, MatrixType& V, int& squarings)
   {
     using std::frexp;
     using std::pow;
     const RealScalar l1norm = arg.cwiseAbs().colwise().sum().maxCoeff();
     squarings = 0;
     if (l1norm < 1.495585217958292e-002) {
       matrix_exp_pade3(arg, U, V);
     } else if (l1norm < 2.539398330063230e-001) {
       matrix_exp_pade5(arg, U, V);
     } else if (l1norm < 9.504178996162932e-001) {
       matrix_exp_pade7(arg, U, V);
     } else if (l1norm < 2.097847961257068e+000) {
       matrix_exp_pade9(arg, U, V);
     } else {
       const RealScalar maxnorm = 5.371920351148152;
       frexp(l1norm / maxnorm, &squarings);
       if (squarings < 0) squarings = 0;
       MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<RealScalar>(squarings));
       matrix_exp_pade13(A, U, V);
     }
   }
 };

 template <typename MatrixType>
 struct matrix_exp_computeUV<MatrixType, long double>
 {
   template <typename ArgType>
   static void run(const ArgType& arg, MatrixType& U, MatrixType& V, int& squarings)
   {
 #if   LDBL_MANT_DIG == 53   // double precision
     matrix_exp_computeUV<MatrixType, double>::run(arg, U, V, squarings);

 #else

     using std::frexp;
     using std::pow;
     const long double l1norm = arg.cwiseAbs().colwise().sum().maxCoeff();
     squarings = 0;

 #if LDBL_MANT_DIG <= 64   // extended precision

     if (l1norm < 4.1968497232266989671e-003L) {
       matrix_exp_pade3(arg, U, V);
     } else if (l1norm < 1.1848116734693823091e-001L) {
       matrix_exp_pade5(arg, U, V);
     } else if (l1norm < 5.5170388480686700274e-001L) {
       matrix_exp_pade7(arg, U, V);
     } else if (l1norm < 1.3759868875587845383e+000L) {
       matrix_exp_pade9(arg, U, V);
     } else {
       const long double maxnorm = 4.0246098906697353063L;
       frexp(l1norm / maxnorm, &squarings);
       if (squarings < 0) squarings = 0;
       MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<long double>(squarings));
       matrix_exp_pade13(A, U, V);
     }

 #elif LDBL_MANT_DIG <= 106  // double-double

     if (l1norm < 3.2787892205607026992947488108213e-005L) {
       matrix_exp_pade3(arg, U, V);
     } else if (l1norm < 6.4467025060072760084130906076332e-003L) {
       matrix_exp_pade5(arg, U, V);
     } else if (l1norm < 6.8988028496595374751374122881143e-002L) {
       matrix_exp_pade7(arg, U, V);
     } else if (l1norm < 2.7339737518502231741495857201670e-001L) {
       matrix_exp_pade9(arg, U, V);
     } else if (l1norm < 1.3203382096514474905666448850278e+000L) {
       matrix_exp_pade13(arg, U, V);
     } else {
       const long double maxnorm = 3.2579440895405400856599663723517L;
       frexp(l1norm / maxnorm, &squarings);
       if (squarings < 0) squarings = 0;
       MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<long double>(squarings));
       matrix_exp_pade17(A, U, V);
     }

 #elif LDBL_MANT_DIG <= 113  // quadruple precision

     if (l1norm < 1.639394610288918690547467954466970e-005L) {
       matrix_exp_pade3(arg, U, V);
     } else if (l1norm < 4.253237712165275566025884344433009e-003L) {
       matrix_exp_pade5(arg, U, V);
     } else if (l1norm < 5.125804063165764409885122032933142e-002L) {
       matrix_exp_pade7(arg, U, V);
     } else if (l1norm < 2.170000765161155195453205651889853e-001L) {
       matrix_exp_pade9(arg, U, V);
     } else if (l1norm < 1.125358383453143065081397882891878e+000L) {
       matrix_exp_pade13(arg, U, V);
     } else {
       const long double maxnorm = 2.884233277829519311757165057717815L;
       frexp(l1norm / maxnorm, &squarings);
       if (squarings < 0) squarings = 0;
       MatrixType A = arg.unaryExpr(MatrixExponentialScalingOp<long double>(squarings));
       matrix_exp_pade17(A, U, V);
     }

 #else

     // this case should be handled in compute()
     eigen_assert(false && "Bug in MatrixExponential");

 #endif
 #endif  // LDBL_MANT_DIG
   }
 };

 template<typename T> struct is_exp_known_type : false_type {};
 template<> struct is_exp_known_type<float> : true_type {};
 template<> struct is_exp_known_type<double> : true_type {};
 #if LDBL_MANT_DIG <= 113
 template<> struct is_exp_known_type<long double> : true_type {};
 #endif

 template <typename ArgType, typename ResultType>
 void matrix_exp_compute(const ArgType& arg, ResultType &result, true_type) // natively supported scalar type
 {
   typedef typename ArgType::PlainObject MatrixType;
   MatrixType U, V;
   int squarings;
   matrix_exp_computeUV<MatrixType>::run(arg, U, V, squarings); // Pade approximant is (U+V) / (-U+V)
   MatrixType numer = U + V;
   MatrixType denom = -U + V;
   result = denom.partialPivLu().solve(numer);
   for (int i=0; i<squarings; i++)
     result *= result;   // undo scaling by repeated squaring
 }


 /* Computes the matrix exponential
  *
  * \param arg    argument of matrix exponential (should be plain object)
  * \param result variable in which result will be stored
  */
 template <typename ArgType, typename ResultType>
 void matrix_exp_compute(const ArgType& arg, ResultType &result, false_type) // default
 {
   typedef typename ArgType::PlainObject MatrixType;
   typedef typename traits<MatrixType>::Scalar Scalar;
   typedef typename NumTraits<Scalar>::Real RealScalar;
   typedef typename std::complex<RealScalar> ComplexScalar;
   result = arg.matrixFunction(internal::stem_function_exp<ComplexScalar>);
 }

 } // end namespace Eigen::internal

 template<typename Derived> struct MatrixExponentialReturnValue
 : public ReturnByValue<MatrixExponentialReturnValue<Derived> >
 {
   public:
     MatrixExponentialReturnValue(const Derived& src) : m_src(src) { }

     template <typename ResultType>
     inline void evalTo(ResultType& result) const
     {
       const typename internal::nested_eval<Derived, 10>::type tmp(m_src);
       internal::matrix_exp_compute(tmp, result, internal::is_exp_known_type<typename Derived::RealScalar>());
     }

     Index rows() const { return m_src.rows(); }
     Index cols() const { return m_src.cols(); }

   protected:
     const typename internal::ref_selector<Derived>::type m_src;
 };

 namespace internal {
 template<typename Derived>
 struct traits<MatrixExponentialReturnValue<Derived> >
 {
   typedef typename Derived::PlainObject ReturnType;
 };
 }

 template <typename Derived>
 const MatrixExponentialReturnValue<Derived> MatrixBase<Derived>::exp() const
 {
   eigen_assert(rows() == cols());
   return MatrixExponentialReturnValue<Derived>(derived());
 }

 } // end namespace Eigen

 #endif // EIGEN_MATRIX_EXPONENTIAL
```

采用的帕德近似实现接口。


# 四、对比分析

Scipy, TensorFlow, Eigen 都使用的是 Pade 近似，PyTorch 采用的是泰勒级数的方式实现。

Pytorch 和 Scipy 二者API设计方式一样，Pytorch采用了最新的方式近似计算矩阵指数，在计算效率上相比 Scipy 更有优势。

TensorFlow 采用了 Python API 的方式实现，基本上是 eigen 实现的转译。在跨平台性能上不如 C++ 算子。

# 五、设计思路与实现方案

## 命名与参数设计

API 设计为 `paddle.linalg.matrix_exp(x, name=None)`。

- `x` 为张量，允许的数据类型是float32, float64 （由于依赖的 paddle 接口限制，此 api 不支持 complex64 和 complex128）。
- `name` 作为可选参数，定义了该操作的名称，其默认值为`None`。

## 底层OP设计

使用 python 实现，不涉及底层 op。

## API实现方案

参考论文 《Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation》 https://www.mdpi.com/2227-7390/7/12/1174

对比 PyTorch 的实现方式与 Pade 近似算法：

> In combination with scaling and squaring, this yields a procedure to compute the matrix exponential up to the desired accuracy at lower computational cost than the standard Padé method for a wide range of matrices. Based on estimates of the form $$\left \| A^k \right \| ^ {1/k}$$ and the use of scaling and squaring, we have presented a modification of our algorithm to reduce overscaling that is still more efficient than state-of-the-art implementations with a slightly lower accuracy for some matrices but higher accuracy for others. The loss in accuracy can be attributed to possible overscalings due to a reduced number of norm estimations compared with Padé methods.

对于一些矩阵在相应的精度下 PyTorch 的算法效率更高，但仍存在一些精度降低的情况。

从之前的调研可以看到，Scipy, TensorFlow, Eigen 都使用的是 Pade 近似，参考资料较多，实现方式较主流。

PyTorch 的实现基于上述论文，发表于 2019 年，参考资料较少。

基于此，本次实现基于 Pade 近似的方式，并且考虑到算子兼容情况，使用 python 的实现方式。

该 API 实现于 `python/paddle/tensor/linalg.py`.

涉及到的 paddle api 主要有：

- Tensor 乘法
- Tensor 加法
- paddle.matmul
- paddle.eye
- paddle.where
- padde.less_than
- paddle.cast
- paddle.pow
- paddle.full
- paddle.max
- paddle.unsqueeze
- paddle.sum
- paddle.log
- paddle.maximum
- paddle.floor
- paddle.isfinite
- paddle.linalg.solve
- paddle.static.nn.cond
- paddle.static.nn.while_loop

实现的主要逻辑伪代码：

``` python
def matrix_exp(x, name=None):
    # step 1. convert to tensor if necessary
    mat_a = paddle.to_tensor(x)

    # step 2. check tensor
    # check dtype
    if dtype not in ['float32', 'float64']:
        raise ValueError(
            f"The input tensor's dtype must be float32 or float64, but got {dtype}"
        )

    # check tensor dim
    if mat_a.ndim < 2:
        raise ValueError('The input tensor must be at least two-dimensional')

    if mat_a.shape[-1] != mat_a.shape[-2]:
        raise ValueError('Last 2 dimensions of the tensor must be square')

    # step 3. compute uv
    l1_norm = paddle.unsqueeze(
        paddle.max(paddle.sum(paddle.abs(mat_a), axis=mat_a.ndim - 2), axis=-1),
        axis=[-1, -2],
    )
    maxnorm = ... # a number const
    squarings = paddle.floor(
        paddle.log(l1_norm / maxnorm)
        / paddle.log(paddle.full((), 2.0, dtype))
    )
    squarings = paddle.maximum(squarings, paddle.zeros_like(squarings))
    u, v = _matrix_uv_func(mat_a, l1_norm, squarings, dtype)

    # step 4. compute result
    if paddle.isfinite(paddle.max(l1_norm)):
      result = paddle.linalg.solve(-u + v, u + v)
    else:
      result = paddle.full(mat_a.shape, np.nan, dtype)

    max_squarings = paddle.max(squarings)
    i = 0
    while i < max_squarings:
      result = paddle.matmul(result, result)
      i += 1

    return result
```

其中，`_matrix_uv_func` 用于计算 u/v，以 float32 为例：

``` python
def _matrix_uv_float32(mat_a, l1_norm, squarings, dtype):
    mat_i = paddle.eye(mat_a.shape[-1], dtype=dtype)
    mat_a2 = paddle.matmul(mat_a, mat_a)
    mat_a4 = paddle.matmul(mat_a2, mat_a2)
    mat_a6 = paddle.matmul(mat_a4, mat_a2)

    u3, v3 = _matrix_exp_pade3(mat_a, mat_i, mat_a2, dtype)
    u5, v5 = _matrix_exp_pade5(mat_a, mat_i, mat_a2, mat_a4, dtype)
    u7, v7 = _matrix_exp_pade7(
        mat_a
        / paddle.cast(
            paddle.pow(paddle.full((), 2.0, dtype), squarings),
            dtype,
        ),
        mat_i,
        mat_a2,
        mat_a4,
        mat_a6,
        dtype,
    )

    if l1_norm < 4.258730016922831e-001:
      u, v = u3, v3
    elif l1_norm < 1.880152677804762e000:
      u, v = u5, v5
    else:
      u, v = u7, v7

    return u, v
```

其中 pade 近似，如 `_matrix_exp_pade3`：

``` python
def _matrix_exp_pade3(mat_a, mat_i, mat_a2, dtype):
    """3rd-order Pade approximant."""
    b = [120.0, 60.0, 12.0]
    b = [paddle.full((), x, dtype) for x in b]
    tmp = mat_a2 + b[1] * mat_i
    mat_u = paddle.matmul(mat_a, tmp)
    mat_v = b[2] * mat_a + b[0] * mat_i
    return mat_u, mat_v
```

另外，由于该API需要考虑动静统一问题，故需要验证其在静态图中能否正常工作。（该工作会在单测中进行）

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
可考虑以下场景：

### 1. 一般场景
* 结果一致性测试。测试对于同一输入和 Scipy 中 matrix_exp API 计算结果的数值的一致性。
* 数据类型测试。选取不同数据类型的输入，测试计算结果的准确性。

### 2. 边界条件
* 当 `x` 为空张量，测试其输出是否空张量且输出张量形状是否正确。

### 3. 异常测试
* 对于参数异常值输入，例如x的不合法值等，应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划
技术可行性：参考同类项目和相似的 API，相关内容参考丰富，无重大难点；

# 八、影响面
为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料

- RFC 文档 [【Hackathon 5th No.36】 为 Paddle 新增 matrix_exp API](https://github.com/PaddlePaddle/community/pull/674)
- 上一期RFC文档[【PaddlePaddle Hackathon 第四期】No.18：为 Paddle 新增 matrix_exp API ](https://github.com/PaddlePaddle/community/pull/523)
- （linalg同类型参考）—— paddle.linalgmatrix_power()
  - https://github.com/PaddlePaddle/Paddle/pull/34667
  - https://github.com/PaddlePaddle/Paddle/pull/40231
- 其他框架 或 建模软件
- [torch.linalg.matrix_exp ‒ PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html#torch-linalg-matrix-exp)
- [scipy.linalg.expm — SciPy v1.11.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)
- [www.tensorflow.org — tf.linalg.expm v2.13.0 ](https://www.tensorflow.org/api_docs/python/tf/linalg/expm)
- [tinyms.primitives — TinyMS alpha documentation](https://tinyms.readthedocs.io/en/latest/tinyms/tinyms.primitives.html#tinyms.primitives.MatrixExp)
- [MatrixExp‒ Wolfram Language Documentation](https://reference.wolfram.com/language/ref/MatrixExp.html)
- [矩阵指数 - MATLAB & Simulink Example - MathWorks 中国](https://ww2.mathworks.cn/help/matlab/math/matrix-exponentials.html)

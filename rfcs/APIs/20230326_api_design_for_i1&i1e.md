# paddle.i1 和 paddle.i1e 设计文档

| API 名称      |         paddle.i1 / paddle.i1e           |
| ------------ | ---------------------------------------- |
| 提交作者    | LyndonKong                               |
| 提交时间    | 2023-03-26                                |
| 版本号      | V1.0                                      |
| 依赖飞桨版本 | develop                                   |
| 文件名      | 20230326_api_design_for_i1&i1e.md      |

# 一、概述

## 1、相关背景

paddle.i1和paddle.Tensor.i1用于计算输入的每个元素第一类一阶修正贝塞尔函数，计算公式为

$$
\text{out}_{i}=\frac{\left(\text { input }_{i}\right)}{2} * \sum_{k=0}^{\infty} \frac{\left(\text{input}_{i}^{2} / 4\right)^{k}}{(k !) *(k+1) !}
$$

paddle.i1e和paddle.Tensor.i1e用于计算输入的每个元素第一类指数缩放的零阶修正贝塞尔函数，计算公式为

$$
\text{out}_{i}=\exp (-|x|) * i 1(x)=\exp (-|x|) * \frac{\left(\text { input }_{i}\right)}{2} * \sum_{k=0}^{\infty} \frac{\left(\text { input }_{i}^{2} / 4\right)^{k}}{(k !) *(k+1) !}
$$

## 2、功能目标

在飞桨中增加 paddle.i1, paddle.Tensor.i1, paddle.i1e 和 paddle.Tensor.i1e API。

## 3、意义

飞桨将支持 paddle.i1, paddle.Tensor.i1, paddle.i1e 和 paddle.Tensor.i1e API。

# 二、飞桨现状

飞桨中还没有 i1/i1e API。为了支持 i1/i1e API，需要新增对应算子前向传播和反向传播的C++实现。


# 三、业内方案调研

i1业内方案
## PyTorch

PyTorch 支持 torch.i1 和 torch.speicial.i1，介绍为
```
Computes the first order modified Bessel function of the first kind for each element of input.
```

### 实现方法
代码实现来自于 Cephes Math 库，将定义域分为$[0, 8]$和$[8, \infty]$两个区间，在每个区间内部分别通过 Chebyshev 多项式展开。下面是具体实现：

#### 前向传播算子

```cpp

/*
 * This function is derived from the implementation of the i1 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};
  return std::make_tuple(coeff, 29);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      9.38153738649577178388E-9f,
      -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f,
      -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f,
      -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f,
      -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f,
      -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f,
      -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f,
      -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f,
      -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
};

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return std::make_tuple(coeff, 25);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  return std::make_tuple(coeff, 7);
};

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    const T out = std::exp(x) * x * chbevl(y, A, len);
    return (_x < T{0.0}) ? -out : out;
  }
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const T out = (std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len)) / std::sqrt(x);
  return (_x < T{0.0}) ? -out : out;
}

```

#### 反向传播算子

```yaml

- name: special_i1(Tensor self) -> Tensor
  self: i1_backward(grad, self, result)
  result: auto_element_wise

```

```cpp
Tensor i1_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result) {
  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1_backward", [&]() {
    // For x = 0, the correct gradient is 0.5,
    // however due to floating point computation we get NaN.
    // So we manually update gradient for x=0
    auto eps = std::numeric_limits<scalar_t>::epsilon();
    auto self_is_not_tiny = self.abs() > eps;

    // Following `where` is needed as `where` computes gradients,
    // even for the part which didn't affect the output.
    // Look at https://github.com/pytorch/pytorch/issues/52248
    // Update if and when this is fixed.
    auto safe_self =
        at::where(self_is_not_tiny, self, at::full({}, eps, self.options()));
    auto gradx = (safe_self.i0() - (result * safe_self.reciprocal()));
    return grad *
        at::where(self_is_not_tiny, gradx, at::full({}, 0.5, self.options()));
  });
}
```


## TensorFlow

TensorFlow支持`tf.math.bessel_i1`，介绍为
```
Computes the Bessel i1 function of x element-wise.
```

### 实现方法

#### 前向传播算子

代码实现来自于 Eigen3 库，下面引入相关代码的位置：

```cpp

template <typename T>
struct bessel_i0 : base<T, Eigen::internal::scalar_bessel_i0_op<T>> {};

```

在 Eigen3 库当中，该函数的实现方式如下

```cpp

template <typename T, typename ScalarType = typename unpacket_traits<T>::type >
struct generic_i1e {
  EIGEN_STATIC_ASSERT((internal::is_same<T, T>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED)

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run(const T&) {
    return ScalarType(0);
  }
};

template <typename T>
struct generic_i1e<T, float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /* i1ef.c
     *
     *  Modified Bessel function of order one,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * float x, y, i1ef();
     *
     * y = i1ef( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order one of the argument.
     *
     * The function is defined as i1(x) = -i exp(-|x|) j1( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0, 30       30000       1.5e-6      1.5e-7
     * See i1().
     *
     */
    const float A[] = {9.38153738649577178388E-9f, -4.44505912879632808065E-8f,
                       2.00329475355213526229E-7f, -8.56872026469545474066E-7f,
                       3.47025130813767847674E-6f, -1.32731636560394358279E-5f,
                       4.78156510755005422638E-5f, -1.61760815825896745588E-4f,
                       5.12285956168575772895E-4f, -1.51357245063125314899E-3f,
                       4.15642294431288815669E-3f, -1.05640848946261981558E-2f,
                       2.47264490306265168283E-2f, -5.29459812080949914269E-2f,
                       1.02643658689847095384E-1f, -1.76416518357834055153E-1f,
                       2.52587186443633654823E-1f};

    const float B[] = {-3.83538038596423702205E-9f, -2.63146884688951950684E-8f,
                       -2.51223623787020892529E-7f, -3.88256480887769039346E-6f,
                       -1.10588938762623716291E-4f, -9.76109749136146840777E-3f,
                       7.78576235018280120474E-1f};


    T y = pabs(x);
    T y_le_eight = pmul(y, internal::pchebevl<T, 17>::run(
        pmadd(pset1<T>(0.5f), y, pset1<T>(-2.0f)), A));
    T y_gt_eight = pmul(
        internal::pchebevl<T, 7>::run(
            psub(pdiv(pset1<T>(32.0f), y),
                 pset1<T>(2.0f)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    y = pselect(pcmp_le(y, pset1<T>(8.0f)), y_le_eight, y_gt_eight);
    return pselect(pcmp_lt(x, pset1<T>(0.0f)), pnegate(y), y);
  }
};

template <typename T>
struct generic_i1e<T, double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /*  i1e.c
     *
     *  Modified Bessel function of order one,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * double x, y, i1e();
     *
     * y = i1e( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order one of the argument.
     *
     * The function is defined as i1(x) = -i exp(-|x|) j1( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0, 30       30000       2.0e-15     2.0e-16
     * See i1().
     *
     */
    const double A[] = {2.77791411276104639959E-18, -2.11142121435816608115E-17,
                        1.55363195773620046921E-16, -1.10559694773538630805E-15,
                        7.60068429473540693410E-15, -5.04218550472791168711E-14,
                        3.22379336594557470981E-13, -1.98397439776494371520E-12,
                        1.17361862988909016308E-11, -6.66348972350202774223E-11,
                        3.62559028155211703701E-10, -1.88724975172282928790E-9,
                        9.38153738649577178388E-9,  -4.44505912879632808065E-8,
                        2.00329475355213526229E-7,  -8.56872026469545474066E-7,
                        3.47025130813767847674E-6,  -1.32731636560394358279E-5,
                        4.78156510755005422638E-5,  -1.61760815825896745588E-4,
                        5.12285956168575772895E-4,  -1.51357245063125314899E-3,
                        4.15642294431288815669E-3,  -1.05640848946261981558E-2,
                        2.47264490306265168283E-2,  -5.29459812080949914269E-2,
                        1.02643658689847095384E-1,  -1.76416518357834055153E-1,
                        2.52587186443633654823E-1};
    const double B[] = {
        7.51729631084210481353E-18,  4.41434832307170791151E-18,
        -4.65030536848935832153E-17, -3.20952592199342395980E-17,
        2.96262899764595013876E-16,  3.30820231092092828324E-16,
        -1.88035477551078244854E-15, -3.81440307243700780478E-15,
        1.04202769841288027642E-14,  4.27244001671195135429E-14,
        -2.10154184277266431302E-14, -4.08355111109219731823E-13,
        -7.19855177624590851209E-13, 2.03562854414708950722E-12,
        1.41258074366137813316E-11,  3.25260358301548823856E-11,
        -1.89749581235054123450E-11, -5.58974346219658380687E-10,
        -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
        -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
        -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
        7.78576235018280120474E-1};
    T y = pabs(x);
    T y_le_eight = pmul(y, internal::pchebevl<T, 29>::run(
        pmadd(pset1<T>(0.5), y, pset1<T>(-2.0)), A));
    T y_gt_eight = pmul(
        internal::pchebevl<T, 25>::run(
            psub(pdiv(pset1<T>(32.0), y),
                 pset1<T>(2.0)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    y = pselect(pcmp_le(y, pset1<T>(8.0)), y_le_eight, y_gt_eight);
    return pselect(pcmp_lt(x, pset1<T>(0.0)), pnegate(y), y);
  }
};

template <typename T, typename ScalarType = typename unpacket_traits<T>::type>
struct generic_i1 {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    return pmul(
        pexp(pabs(x)),
        generic_i1e<T, ScalarType>::run(x));
  }
};

```

#### 反向传播算子

```python

@ops.RegisterGradient("BesselI1")
def _BesselI1Grad(op, grad):
  """Compute gradient of bessel_i1(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # For x = 0, the correct gradient is 1.0.
    # However, the main branch gives NaN because of the division by x, so
    # we impute the gradient manually.
    # An alternative solution is to express the gradient via bessel_i0 and
    # bessel_i2, but the latter is not yet implemented in Eigen.
    dy_dx = array_ops.where_v2(
        math_ops.equal(x, 0.), math_ops.cast(1., x.dtype),
        special_math_ops.bessel_i0(x) - math_ops.div(y, x))
    return grad * dy_dx

```


## Scipy

Scipy当中支持通过`scipy.speicial.i1(x, output = None)`方式调用，介绍为
```
Modified Bessel function of order 1.
```

### 实现方法

代码实现来自于 Cephes Math 库，相关说明在[参考链接](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1.html#scipy.special.i1)中，这里不再列出 Cephes Math 库当中的源码。

i1e业内方案
## PyTorch

PyTorch 支持 torch.i1e 和 torch.speicial.i1e，介绍为
```
Computes the exponentially scaled first order modified Bessel function of the first kind for each element of input.
```

### 实现方法
代码实现来自于 Cephes Math 库，将定义域分为$[0, 8]$和$[8, \infty]$两个区间，f在每个区间内部分别通过 Chebyshev 多项式展开。下面是具体实现：

#### 前向传播算子

```cpp

/*
 * This function is derived from the implementation of the i1e function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the exponentially scaled first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};
  return std::make_tuple(coeff, 29);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      9.38153738649577178388E-9f,
      -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f,
      -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f,
      -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f,
      -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f,
      -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f,
      -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f,
      -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f,
      -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
};

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return std::make_tuple(coeff, 25);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  return std::make_tuple(coeff, 7);
};

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1e(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    const T out = chbevl(y, A, len) * x;
    return (_x < T{0.0}) ? -out : out;
  }
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const auto out = chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
  return (_x < T{0.0}) ? -out : out;
}

```

#### 反向传播算子

```yaml

- name: special_i1e(Tensor self) -> Tensor
  self: i1e_backward(grad, self, result)
  result: auto_element_wise

```

```cpp

Tensor i1e_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result) {
  return AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1e_backward", [&]() {
    // For x = 0, the correct gradient is 0.5,
    // however due to floating point computation we get NaN.
    // So we manually update gradient for x=0
    auto eps = std::numeric_limits<scalar_t>::epsilon();
    auto self_is_not_tiny = self.abs() > eps;

    // Following `where` is needed as `where` computes gradients,
    // even for the part which didn't affect the output.
    // Look at https://github.com/pytorch/pytorch/issues/52248
    // Update if and when this is fixed.
    auto safe_self =
        at::where(self_is_not_tiny, self, at::full({}, eps, self.options()));
    auto gradx =
        (at::special_i0e(safe_self) -
         result * (safe_self.sgn() + safe_self.reciprocal()));
    return grad *
        at::where(self_is_not_tiny, gradx, at::full({}, 0.5, self.options()));
  });
}

```

## TensorFlow

TensorFlow支持`tf.math.bessel_i1e`，介绍为
```
Computes the Bessel i1e function of x element-wise.
```

### 实现方法

#### 正向传播算子

代码实现来自于 Eigen3 库，下面引入相关代码的位置：

```cpp

template <typename T>
struct bessel_i1e : base<T, Eigen::internal::scalar_bessel_i1e_op<T>> {};

```

在 Eigen3 库当中，该函数的实现方式如下

```cpp

template <typename T, typename ScalarType = typename unpacket_traits<T>::type >
struct generic_i1e {
  EIGEN_STATIC_ASSERT((internal::is_same<T, T>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED)

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run(const T&) {
    return ScalarType(0);
  }
};

template <typename T>
struct generic_i1e<T, float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /* i1ef.c
     *
     *  Modified Bessel function of order one,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * float x, y, i1ef();
     *
     * y = i1ef( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order one of the argument.
     *
     * The function is defined as i1(x) = -i exp(-|x|) j1( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0, 30       30000       1.5e-6      1.5e-7
     * See i1().
     *
     */
    const float A[] = {9.38153738649577178388E-9f, -4.44505912879632808065E-8f,
                       2.00329475355213526229E-7f, -8.56872026469545474066E-7f,
                       3.47025130813767847674E-6f, -1.32731636560394358279E-5f,
                       4.78156510755005422638E-5f, -1.61760815825896745588E-4f,
                       5.12285956168575772895E-4f, -1.51357245063125314899E-3f,
                       4.15642294431288815669E-3f, -1.05640848946261981558E-2f,
                       2.47264490306265168283E-2f, -5.29459812080949914269E-2f,
                       1.02643658689847095384E-1f, -1.76416518357834055153E-1f,
                       2.52587186443633654823E-1f};

    const float B[] = {-3.83538038596423702205E-9f, -2.63146884688951950684E-8f,
                       -2.51223623787020892529E-7f, -3.88256480887769039346E-6f,
                       -1.10588938762623716291E-4f, -9.76109749136146840777E-3f,
                       7.78576235018280120474E-1f};


    T y = pabs(x);
    T y_le_eight = pmul(y, internal::pchebevl<T, 17>::run(
        pmadd(pset1<T>(0.5f), y, pset1<T>(-2.0f)), A));
    T y_gt_eight = pmul(
        internal::pchebevl<T, 7>::run(
            psub(pdiv(pset1<T>(32.0f), y),
                 pset1<T>(2.0f)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    y = pselect(pcmp_le(y, pset1<T>(8.0f)), y_le_eight, y_gt_eight);
    return pselect(pcmp_lt(x, pset1<T>(0.0f)), pnegate(y), y);
  }
};

template <typename T>
struct generic_i1e<T, double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /*  i1e.c
     *
     *  Modified Bessel function of order one,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * double x, y, i1e();
     *
     * y = i1e( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order one of the argument.
     *
     * The function is defined as i1(x) = -i exp(-|x|) j1( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0, 30       30000       2.0e-15     2.0e-16
     * See i1().
     *
     */
    const double A[] = {2.77791411276104639959E-18, -2.11142121435816608115E-17,
                        1.55363195773620046921E-16, -1.10559694773538630805E-15,
                        7.60068429473540693410E-15, -5.04218550472791168711E-14,
                        3.22379336594557470981E-13, -1.98397439776494371520E-12,
                        1.17361862988909016308E-11, -6.66348972350202774223E-11,
                        3.62559028155211703701E-10, -1.88724975172282928790E-9,
                        9.38153738649577178388E-9,  -4.44505912879632808065E-8,
                        2.00329475355213526229E-7,  -8.56872026469545474066E-7,
                        3.47025130813767847674E-6,  -1.32731636560394358279E-5,
                        4.78156510755005422638E-5,  -1.61760815825896745588E-4,
                        5.12285956168575772895E-4,  -1.51357245063125314899E-3,
                        4.15642294431288815669E-3,  -1.05640848946261981558E-2,
                        2.47264490306265168283E-2,  -5.29459812080949914269E-2,
                        1.02643658689847095384E-1,  -1.76416518357834055153E-1,
                        2.52587186443633654823E-1};
    const double B[] = {
        7.51729631084210481353E-18,  4.41434832307170791151E-18,
        -4.65030536848935832153E-17, -3.20952592199342395980E-17,
        2.96262899764595013876E-16,  3.30820231092092828324E-16,
        -1.88035477551078244854E-15, -3.81440307243700780478E-15,
        1.04202769841288027642E-14,  4.27244001671195135429E-14,
        -2.10154184277266431302E-14, -4.08355111109219731823E-13,
        -7.19855177624590851209E-13, 2.03562854414708950722E-12,
        1.41258074366137813316E-11,  3.25260358301548823856E-11,
        -1.89749581235054123450E-11, -5.58974346219658380687E-10,
        -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
        -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
        -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
        7.78576235018280120474E-1};
    T y = pabs(x);
    T y_le_eight = pmul(y, internal::pchebevl<T, 29>::run(
        pmadd(pset1<T>(0.5), y, pset1<T>(-2.0)), A));
    T y_gt_eight = pmul(
        internal::pchebevl<T, 25>::run(
            psub(pdiv(pset1<T>(32.0), y),
                 pset1<T>(2.0)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    y = pselect(pcmp_le(y, pset1<T>(8.0)), y_le_eight, y_gt_eight);
    return pselect(pcmp_lt(x, pset1<T>(0.0)), pnegate(y), y);
  }
};

```

#### 反向传播算子

```python

@ops.RegisterGradient("BesselI1e")
def _BesselI1eGrad(op, grad):
  """Compute gradient of bessel_i1e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    # For x = 0, the correct gradient is 0.5.
    # However, the main branch gives NaN because of the division by x, so
    # we impute the gradient manually.
    # An alternative solution is to express the gradient via bessel_i0e and
    # bessel_i2e, but the latter is not yet implemented in Eigen.
    dy_dx = array_ops.where_v2(
        math_ops.equal(x, 0.), math_ops.cast(0.5, x.dtype),
        special_math_ops.bessel_i0e(x) - y *
        (math_ops.sign(x) + math_ops.reciprocal(x)))
    return grad * dy_dx

```

## Scipy

Scipy当中支持通过`scipy.speicial.i1e(x, output = None)`方式调用，介绍为
```
Exponentially scaled modified Bessel function of order 1
```

### 实现方法

代码实现来自于 Cephes Math 库，相关说明在[参考链接](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i1e.html)中，这里不再列出 Cephes Math 库当中的源码。

# 四、对比分析

上述三个方案实际相同，为了将当前 paddle 算子的实现和 Eigen 解耦，将参考 Pytorch 独立实现 C++ API。
# 五、设计思路与实现方案

## 命名与参数设计

共添加以下四个 API：

`paddle.i1(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

`paddle.Tensor.i1(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

`paddle.i1e(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

和

`paddle.Tensor.i1e(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

参数与文档要求进行对齐。

## 实现方案

i1
### 前向传播算子

参考 Cephes Math 和 Eigen3 的处理方式，需要调用 `paddle.i1e` ，我们将会随同本方案一同提交 `paddle.i1e` 的API设计方案和相关实现。
1. 计算
   1. 对输入的张量求绝对值求指数 `(paddle.exp(paddle.abs(x)))` 作为 `paddle.i1e` 的输入参数。
   2. 调用 `paddle.i1e` ，根据输入的数据类型 `dtype` 动态派发到不同的函数中，对于 float64 类型的输入使用更多的项进行近似。


### 反向传播算子

反向传播的计算过程将会依赖 `paddle.i0` 的前向传播过程，我们将会随同本方案一同提交 `paddle.i0` 的API设计方案和相关实现。偏导数的计算公式为$i0(\text{input}) - \frac{\text{output}}{\text{input
}}$
1. 计算
  1. 检查输入张量，为值为0的元素手动设置梯度。
  2. 返回 `grad * paddle.i0(x) - paddle.div(output, input)`

i1e
### 前向传播算子

参考 Cephes Math 和 Eigen3 的处理方式，将输入向量的每一个元素
1. 计算
   1. 根据输入的数据类型 `dtype` 动态派发到不同的函数中，对于 float64 类型的输入使用更多的项进行近似。
   2. 对输入的张量逐元素求绝对值。
   3. 将绝对值和分割点(在当前实现中为8)进行比较，对不同区间范围的元素使用不同参数的 Cheysheve 多项式进行近似。


### 反向传播算子

反向传播的计算过程将会依赖 `paddle.i0e` 的前向传播过程，我们将会随同本方案一同提交 `paddle.i0e` 的API设计方案和相关实现。偏导数的计算公式为$i0e(\text{input}) - (sign(\text{input}) + \frac{1}{x}) * \text{output}$
1. 计算
  1. 检查输入张量，为值为0的元素手动设置梯度。
  2. 返回 `grad * (addle.i1(input) - (paddel.sign(input) + paddle.reciprocal(input)) * output)`

# 六、测试和验收的考量

对比scipy.speicial.i1/scipy.special.i1e的实现版本验证进行了前向传播算子的测验和验收。通过scipy.special.i0和scipy.special.i0e自行实现了反向传播函数，并与参考方案对比验证了反向传播函数实现的正确性，在此基础上进行了反向传播算子的测验和验收。

1. 结果正确性:
  - 前向和反向传播梯度计算的正确性；
  - 对不同dtype的输入数据进行计算精度检验(float32, float64)
  - 对不同范围内的输入数据进行计算精度检验($[0, 8]$, $[8, \infty]$)
  - 覆盖静态图和动态图测试场景

2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

# 七、可行性分析和排期规划

技术可行性：参考同类项目和paddle中已有的C++算子，无重大难点。已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面

为独立的新增API，对其他模块没有影响。


# 附件及参考资料

[PyTorch实现](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Math.h)
[TensorFlow API文档](https://www.tensorflow.org/api_docs/python/tf/math/bessel_i1)
[Rational Approximations for the Modified Bessel Function of the First Kind – I1(x) for Computations with Double Precision](https://www.bing.com/search?q=%22Rational+Approximations+for+the+Modified+Bessel+Function+of+the+First+Kind+%0D%0A%23+++++-+I1%28x%29+for+Computations+with+Double+Precision%22+by+Pavel+Holoborodko&qs=n&form=QBRE&sp=-1&lq=1&pq=%22rational+approximations+for+the+modified+bessel+function+of+the+first+kind+%23+-+i1%28x%29+for+computations+with+double+precision%22+by+pavel+holoborodko&sc=0-146&sk=&cvid=2507FDB609B24E39BEA10DE5D3482BD8&ghsh=0&ghacc=0&ghpl=)
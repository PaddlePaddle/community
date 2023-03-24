# PoissonNLLLoss 设计文档

| API名称                                                      | i0                                |
| ------------------------------------------------------------ |------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | LyndonKong                                      |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-24                                     |
| 版本号                                                       | V1.0                                           |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                        |
| 文件名                                                       | 20230324_api_design_for_i0.md<br> |


# 一、概述

## 1、相关背景

paddle.i0和paddle.Tensor.i0用于计算输入的每个元素第一类零阶修正贝塞尔函数，计算公式为

$$
\text{out}_i = I_{0}(\text{input}_i) = \sum\limits_{k=0}^\infty \frac{(\text{input}_i^2/4)^k}{(k!)^2}
$$

## 2、功能目标

在飞桨中增加 paddle.i0 和 paddle.Tensor.i0 API。

## 3、意义

飞桨将支持 paddle.i0 和 paddle.Tensor.i0 API。

# 二、飞桨现状

飞桨中还没有 i0 API。为了支持 i0 API，需要新增对应算子前向传播和反向传播的C++实现。


# 三、业内方案调研

## PyTorch

PyTorch 支持 torch.i0 和 torch.speicial.i0，介绍为
```
Computes the zeroth order modified Bessel function of the first kind for each element of input.
```

### 实现方法
代码实现来自于 Cephes Math 库，将定义域分为$[0, 8]$和$[8, \infty]$两个区间，在每个区间内部分别通过 Chebyshev 多项式展开。下面是具体实现：

#### 前向传播算子

```cpp

/*
 * This function is derived from the implementation of the i0 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the zeroth order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
static inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_A() {
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static const T coeff[] = {
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};
  return std::make_tuple(coeff, 30);
};

template <typename T>
static inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return std::make_tuple(coeff, 25);
};

template <typename T>
static inline typename std::enable_ifi0<std::is_floating_point<T>::value, T>::type
calc_i0(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    return static_cast<T>(std::exp(x) * chbevl(y, A, len));
  }
  auto coeff_pair = chebyshev_coefficients_i0e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  return std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
}

```

#### 反向传播算子

```yaml

- name: i0(Tensor self) -> Tensor
  self: grad * at::special_i1(self)
  result: auto_element_wise

```

## TensorFlow

TensorFlow支持`tf.math.bessel_i0`，介绍为
```
Computes the Bessel i0 function of x element-wise.
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

template <typename T, typename ScalarType = typename unpacket_traits<T>::type>
struct generic_i0e {
  EIGEN_STATIC_ASSERT((internal::is_same<T, T>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED)

  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T run(const T&) {
    return ScalarType(0);
  }
};

template <typename T>
struct generic_i0e<T, float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /*  i0ef.c
     *
     *  Modified Bessel function of order zero,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * float x, y, i0ef();
     *
     * y = i0ef( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order zero of the argument.
     *
     * The function is defined as i0e(x) = exp(-|x|) j0( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        100000      3.7e-7      7.0e-8
     * See i0f().
     *
     */

    const float A[] = {-1.30002500998624804212E-8f, 6.04699502254191894932E-8f,
                       -2.67079385394061173391E-7f, 1.11738753912010371815E-6f,
                       -4.41673835845875056359E-6f, 1.64484480707288970893E-5f,
                       -5.75419501008210370398E-5f, 1.88502885095841655729E-4f,
                       -5.76375574538582365885E-4f, 1.63947561694133579842E-3f,
                       -4.32430999505057594430E-3f, 1.05464603945949983183E-2f,
                       -2.37374148058994688156E-2f, 4.93052842396707084878E-2f,
                       -9.49010970480476444210E-2f, 1.71620901522208775349E-1f,
                       -3.04682672343198398683E-1f, 6.76795274409476084995E-1f};

    const float B[] = {3.39623202570838634515E-9f, 2.26666899049817806459E-8f,
                       2.04891858946906374183E-7f, 2.89137052083475648297E-6f,
                       6.88975834691682398426E-5f, 3.36911647825569408990E-3f,
                       8.04490411014108831608E-1f};
    T y = pabs(x);
    T y_le_eight = internal::pchebevl<T, 18>::run(
        pmadd(pset1<T>(0.5f), y, pset1<T>(-2.0f)), A);
    T y_gt_eight = pmul(
        internal::pchebevl<T, 7>::run(
            psub(pdiv(pset1<T>(32.0f), y), pset1<T>(2.0f)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    return pselect(pcmp_le(y, pset1<T>(8.0f)), y_le_eight, y_gt_eight);
  }
};

template <typename T>
struct generic_i0e<T, double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    /*  i0e.c
     *
     *  Modified Bessel function of order zero,
     *  exponentially scaled
     *
     *
     *
     * SYNOPSIS:
     *
     * double x, y, i0e();
     *
     * y = i0e( x );
     *
     *
     *
     * DESCRIPTION:
     *
     * Returns exponentially scaled modified Bessel function
     * of order zero of the argument.
     *
     * The function is defined as i0e(x) = exp(-|x|) j0( ix ).
     *
     *
     *
     * ACCURACY:
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       5.4e-16     1.2e-16
     * See i0().
     *
     */

    const double A[] = {-4.41534164647933937950E-18, 3.33079451882223809783E-17,
                        -2.43127984654795469359E-16, 1.71539128555513303061E-15,
                        -1.16853328779934516808E-14, 7.67618549860493561688E-14,
                        -4.85644678311192946090E-13, 2.95505266312963983461E-12,
                        -1.72682629144155570723E-11, 9.67580903537323691224E-11,
                        -5.18979560163526290666E-10, 2.65982372468238665035E-9,
                        -1.30002500998624804212E-8,  6.04699502254191894932E-8,
                        -2.67079385394061173391E-7,  1.11738753912010371815E-6,
                        -4.41673835845875056359E-6,  1.64484480707288970893E-5,
                        -5.75419501008210370398E-5,  1.88502885095841655729E-4,
                        -5.76375574538582365885E-4,  1.63947561694133579842E-3,
                        -4.32430999505057594430E-3,  1.05464603945949983183E-2,
                        -2.37374148058994688156E-2,  4.93052842396707084878E-2,
                        -9.49010970480476444210E-2,  1.71620901522208775349E-1,
                        -3.04682672343198398683E-1,  6.76795274409476084995E-1};
    const double B[] = {
        -7.23318048787475395456E-18, -4.83050448594418207126E-18,
        4.46562142029675999901E-17,  3.46122286769746109310E-17,
        -2.82762398051658348494E-16, -3.42548561967721913462E-16,
        1.77256013305652638360E-15,  3.81168066935262242075E-15,
        -9.55484669882830764870E-15, -4.15056934728722208663E-14,
        1.54008621752140982691E-14,  3.85277838274214270114E-13,
        7.18012445138366623367E-13,  -1.79417853150680611778E-12,
        -1.32158118404477131188E-11, -3.14991652796324136454E-11,
        1.18891471078464383424E-11,  4.94060238822496958910E-10,
        3.39623202570838634515E-9,   2.26666899049817806459E-8,
        2.04891858946906374183E-7,   2.89137052083475648297E-6,
        6.88975834691682398426E-5,   3.36911647825569408990E-3,
        8.04490411014108831608E-1};
    T y = pabs(x);
    T y_le_eight = internal::pchebevl<T, 30>::run(
        pmadd(pset1<T>(0.5), y, pset1<T>(-2.0)), A);
    T y_gt_eight = pmul(
        internal::pchebevl<T, 25>::run(
            psub(pdiv(pset1<T>(32.0), y), pset1<T>(2.0)), B),
        prsqrt(y));
    // TODO: Perhaps instead check whether all packet elements are in
    // [-8, 8] and evaluate a branch based off of that. It's possible
    // in practice most elements are in this region.
    return pselect(pcmp_le(y, pset1<T>(8.0)), y_le_eight, y_gt_eight);
  }
};


template <typename T, typename ScalarType = typename unpacket_traits<T>::type>
struct generic_i0 {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T& x) {
    return pmul(
        pexp(pabs(x)),
        generic_i0e<T, ScalarType>::run(x));
  }
};

```

#### 反向传播算子

```python

@ops.RegisterGradient("BesselI0")
def _BesselI0Grad(op, grad):
  """Compute gradient of bessel_i0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = special_math_ops.bessel_i1(x)
    return grad * partial_x

```


## Scipy

Scipy当中支持通过`scipy.speicial.i0(x, output = None)`方式调用，介绍为
```
Modified Bessel function of order 0.
```

### 实现方法

代码实现来自于 Cephes Math 库，相关说明在[参考链接](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0.html#scipy.special.i0)中，这里不再列出 Cephes Math 库当中的源码。

# 四、对比分析

上述三个方案实际相同，在TensorFlow的实现版本中对于double类型的输入在 Chebyshev 多项式中使用了更多项目来近似，由于当前Paddle的部分算子当中已经引入了 Eigen3，在当前的版本当中将参考 TensorFlow 的版本直接调用 Eigen3 中的实现对齐主流方案当中的计算精度。

# 五、设计思路与实现方案

## 命名与参数设计

共添加以下两个 API：

`paddle.i0(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

和

`paddle.Tensor.i0(input, name = None) -> Tensor:`
 - Input(Tensor): 输入向量，形状为`(*)`，`*`表示任何数量的额外维度。
 - Name: 操作的名称，默认为None。

参数与文档要求进行对齐。

## 实现方案

### 前向传播算子

参考 Cephes Math 和 Eigen3 的处理方式，需要调用 `paddle.i0e` ，我们将会随同本方案一同提交 `paddle.i0e` 的API设计方案和相关实现。
1. 计算
   1. 对输入的张量求绝对值求指数 `(paddle.exp(paddle.abs(x)))` 作为 `paddle.i0e` 的输入参数。
   2. 调用 `paddle.i0e` ，根据输入的数据类型 `dtype` 动态派发到不同的函数中，对于 float64 类型的输入使用更多的项进行近似。


### 反向传播算子

反向传播的计算过程将会依赖 `paddle.i1` 的前向传播过程，我们将会随同本方案一同提交 `paddle.i1` 的API设计方案和相关实现。
1. 计算
  1. 返回 `grad * paddle.i1(input)`


# 六、测试和验收的考量

对比scipy.speicial.i0的实现版本验证进行了前向传播算子的测验和验收。通过scipy.special.i1自行实现了反向传播函数，并与参考方案对比验证了反向传播函数实现的正确性，在此基础上进行了反向传播算子的测验和验收。

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
[Rational Approximations for the Modified Bessel Function of the First Kind – I0(x) for Computations with Double Precision](https://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision/#:~:text=In%20this%20post%20we%20will%20study%20properties%20of,new%20approximation%20which%20delivers%20the%20lowest%20relative%20error.)
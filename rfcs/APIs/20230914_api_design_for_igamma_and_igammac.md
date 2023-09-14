# igamma 和 igammac 设计文档
| API 名称     | igamma / igammac            |
| ------------ | --------------------------------- |
| 提交作者     | zrr1999                       |
| 提交时间     | 2023-09-14                        |
| 版本号       | V1.0                              |
| 依赖飞桨版本  | develop                           |
| 文件名       | 20230914_api_design_for_igamma_and_igammac.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持随机分布生成相关 API，Paddle 需要扩充 API `paddle.igamma`, `paddle.igammac`。

## 2、功能目标
新增 paddle.igamma /igammac API，即实现(上)不完全伽马函数和补(下)不完全伽马函数的 API。
这两个函数的定义如下：
$$ \Gamma(a, x) = \int_x^{\infty} t^{a-1} e^{-t} dt $$
$$ \gamma(a, x) = \int_0^x t^{a-1} e^{-t} dt $$

相应的 API 需要输入两个参数 `input` 与 `other`，对应上式的 $a$ 和 $x$；

## 3、意义

为 Paddle 增加(上)不完全伽马函数和补(下)不完全伽马函数，丰富 `paddle` 的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `igamma` 和 `igammac` API，无法方便地计算(上)不完全伽马函数和补(下)不完全伽马函数的数值，以及 inplace 的方式修改输入 `x`。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `torch.igamma(input, other, *, out=None)`和`torch.igammac(input, other, *, out=None)` 的 API，以及相应inplace版本。
PyTorch 中有 `torch.Tensor.igamma(other)` 和 `torch.Tensor.igammac(other)` 的 API，以及相应inplace版本。

因为 PyTorch 中这些 API 的实际计算逻辑相似性较大，因此下文的分析均以 igammac 为例。

在 PyTorch (aten/src/ATen/native/Math.h)中，不完全伽马函数的核心计算逻辑是 `calc_igammac`/`calc_igammacc` 函数，
然后针对不同架构，进行了不同的并行化操作，核心计算逻辑代码如下
```cpp
template <typename scalar_t>
static inline scalar_t calc_igammac(scalar_t a, scalar_t x) {
  /* the calculation of the regularized upper incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.4 [igam1])
   * - if x > 1.1 and x < a, using the substraction from the regularized lower
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (5)
   */
  scalar_t absxma_a;

  static scalar_t SMALL = 20.0;
  static scalar_t LARGE = 200.0;
  static scalar_t SMALLRATIO = 0.3;
  static scalar_t LARGERATIO = 4.5;

  // note that in SciPy, a and x are non-negative, with exclusive 0s (i.e.,
  // at most 1 of them can be 0), where igammac(0, x) = 0.0 iff x > 0.
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 0.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 1.0;
  }
  else if (std::isinf(a)) {
    if (std::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 1.0;
  }
  else if (std::isinf(x)) {
    return 0.0;
  }

  absxma_a = std::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / std::sqrt(a))) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }

  if (x > 1.1) {
    if (x < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_continued_fraction(a, x);
    }
  }
  else if (x <= 0.5) {
    if (-0.4 / std::log(x) < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
  else {
    if (x * 1.1 < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
}
```

针对一般 CPU 的并行化处理，主要是给`Vectorized`结构体添加一个新方法，代码如下(aten/src/ATen/cpu/vec/vec256/vec256_float.h)
```cpp
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
```

针对 CUDA 的并行化处理，代码如下(aten/src/ATen/native/cuda/IGammaKernel.cu)
```cpp
template <typename scalar_t>
__noinline__ __host__ __device__ scalar_t calc_igammac(scalar_t a, scalar_t x) {
  /* the calculation of the regularized upper incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.4 [igam1])
   * - if x > 1.1 and x < a, using the substraction from the regularized lower
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (5)
   */

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
  accscalar_t absxma_a;

  static const accscalar_t SMALL = 20.0;
  static const accscalar_t LARGE = 200.0;
  static const accscalar_t SMALLRATIO = 0.3;
  static const accscalar_t LARGERATIO = 4.5;

  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<accscalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 0.0;
    }
    else {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 1.0;
  }
  else if (::isinf(static_cast<accscalar_t>(a))) {
    if (::isinf(static_cast<accscalar_t>(x))) {
      return std::numeric_limits<accscalar_t>::quiet_NaN();
    }
    return 1.0;
  }
  else if (::isinf(static_cast<accscalar_t>(x))) {
    return 0.0;
  }

  absxma_a = ::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / ::sqrt(a))) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }

  if (x > 1.1) {
    if (x < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_continued_fraction(a, x);
    }
  }
  else if (x <= 0.5) {
    if (-0.4 / ::log(x) < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
  else {
    if (x * 1.1 < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
}


```

## Scipy

Scipy 中有 `scipy.special.gammainc(a, x, dps=50, maxterms=10**8)`和`scipy.special.gammaincc(a, x, dps=50, maxterms=10**8)` 的 API。

在 Scipy (scipy/special/_precompute/gammainc_data.py)中，
gammainc 通过超几何函数计算，代码如下
```py
def gammainc(a, x, dps=50, maxterms=10**8):
    """Compute gammainc exactly like mpmath does but allow for more
    summands in hypercomb. See

    mpmath/functions/expintegrals.py#L134

    in the mpmath github repository.

    """
    with mp.workdps(dps):
        z, a, b = mp.mpf(a), mp.mpf(x), mp.mpf(x)
        G = [z]
        negb = mp.fneg(b, exact=True)

        def h(z):
            T1 = [mp.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
            return (T1,)

        res = mp.hypercomb(h, [z], maxterms=maxterms)
        return mpf2float(res)
```


## TensorFlow
TensorFlow 中有 `Igamma(a: XlaOp, x: XlaOp)`和`Igammac(a: XlaOp, x: XlaOp)` 的 API。

TensorFlow 会转换成 XLA，最后 XLA 的实现代码如下：
```cpp

XlaOp Igamma(XlaOp a, XlaOp x) {
  auto& b = *a.builder();
  auto doit = [&b](XlaOp a, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp is_nan = Or(IsNan(a), IsNan(x));
    XlaOp x_is_zero = Eq(x, ScalarLike(x, 0));
    XlaOp x_is_infinity =
        Eq(x, ScalarLike(x, std::numeric_limits<float>::infinity()));
    XlaOp domain_error = Or(Lt(x, ScalarLike(x, 0)), Le(a, ScalarLike(a, 0)));
    XlaOp use_igammac = And(Gt(x, ScalarLike(x, 1)), Gt(x, a));
    XlaOp ax = a * Log(x) - x - Lgamma(a);
    XlaOp underflow = Lt(ax, -Log(MaxFiniteValue(&b, type)));
    ax = Exp(ax);
    XlaOp enabled = Not(Or(Or(Or(x_is_zero, domain_error), underflow), is_nan));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    XlaOp output = Select(
        use_igammac,
        ScalarLike(a, 1) - IgammacContinuedFraction<VALUE>(
                               ax, x, a, And(enabled, use_igammac), type),
        IgammaSeries<VALUE>(ax, x, a, And(enabled, Not(use_igammac)), type));
    output = Select(x_is_zero, ZerosLike(output), output);
    output = Select(x_is_infinity, FullLike(output, 1), output);
    output = Select(Or(domain_error, is_nan), FullLike(a, nan), output);
    return output;
  };
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto a_shape, b.GetShape(a));
    TF_ASSIGN_OR_RETURN(auto x_shape, b.GetShape(x));
    if (a_shape != x_shape) {
      return InvalidArgument(
          "Arguments to Igamma must have equal shapes and types; got %s and %s",
          a_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Igamma", a));
    PrimitiveType a_x_type = a_shape.element_type();
    bool needs_upcast = false;
    for (PrimitiveType type :
         {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ}) {
      if (a_shape.element_type() == type) {
        needs_upcast = true;
        break;
      }
    }

    if (needs_upcast) {
      a = ConvertElementType(a, F32);
      x = ConvertElementType(x, F32);
      a_x_type = F32;
    }
    XlaOp result = doit(a, x, a_x_type);
    if (needs_upcast) {
      result = ConvertElementType(result, a_shape.element_type());
    }
    return result;
  });
}

```

# 四、对比分析

## 共同点
- 都有提供对 Python 的调用接口。
- 均支持 tensor 的输入。

## 不同点

- PyTorch 是使用 C++ 独立编写的计算逻辑。
- Scipy 是使用超几何函数计算。
- Tensorflow 是通过转换为 XLA 再进行计算。

# 五、设计思路与实现方案

## 命名与参数设计

添加 Python API

```python
paddle.igamma(
    inout: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igammac(
    inout: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igamma_(
    inout: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igammac_(
    inout: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.Tensor.igamma(
    other: Tensor
)
```

```python
paddle.Tensor.igammac(
    other: Tensor
)
```

```python
paddle.Tensor.igamma_(
    other: Tensor
)
```

```python
paddle.Tensor.igammac_(
    other: Tensor
)
```

## 底层OP设计

不涉及

## API实现方案

该 API 实现于 `python/paddle/tensor/manipulation.py`。

### igamma
参考 PyTorch 的实现，使用 C++ 独立编写的计算逻辑。

### igammac


# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 scipy 作为参考标准
- 对不同 dtype 的输入数据 `x` 进行计算精度检验 (float32, float64)
- 输入输出的容错性与错误提示信息
- 输出 Dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景

# 七、可行性分析和排期规划

方案主要依赖现有原理实现。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响

# 名词解释
gammainc 是 igamma 的另一种写法，gammaincc 是 igammac 的另一种写法。

# 附件及参考资料

- [torch.igamma](https://pytorch.org/docs/stable/special.html#torch.special.gammainc)
- [torch.igammac](https://pytorch.org/docs/stable/special.html#torch.special.gammaincc)
- [scipy](https://github.com/scipy/scipy)
- [tensorflow](https://github.com/tensorflow/tensorflow)

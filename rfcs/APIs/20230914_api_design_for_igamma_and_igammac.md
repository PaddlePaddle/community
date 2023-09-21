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

为了提升飞桨 API 丰富度，支持随机分布生成相关 API，Paddle 需要扩充 API `paddle.igamma`, `paddle.igammac`, `paddle.igamma_`, `paddle.igammac_`。

## 2、功能目标
新增 `paddle.igamma`, `paddle.igammac`, `paddle.igamma_`, `paddle.igammac_` API，即实现[上不完全伽马函数和下不完全伽马](https://wuli.wiki/online/IncGam.html)函数的 API。
这两个函数的定义如下：

$\Gamma(a, x) = \int_x^{\infty} t^{a-1} e^{-t} dt $

$\gamma(a, x) = \int_0^x t^{a-1} e^{-t} dt $

上不完全伽马函数 $\Gamma(a,x)$ 的定义域为 $a>0$， $x \geq 0$，值域为 $(0,\Gamma(a)]$。
下不完全伽马函数 $\gamma(a,x)$ 的定义域为 $a>0$， $x \geq 0$，值域为 $[0,\Gamma(a))$，其中 $\Gamma(a)$ 是伽马函数的值。

相应的 API 需要输入两个参数 `input` 与 `other`，对应上式的 $a$ 和 $x$；

## 3、意义

为 Paddle 增加上不完全伽马函数和下不完全伽马函数，丰富 `paddle` 的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `paddle.igamma`, `paddle.igammac`, `paddle.igamma_`, `paddle.igammac_` API，无法方便地计算上不完全伽马函数和下不完全伽马函数的数值，以及 inplace 的方式修改输入 `x`。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `torch.igamma(input, other, *, out=None)`和`torch.igammac(input, other, *, out=None)` 的 API，以及相应inplace版本。
PyTorch 中有 `torch.Tensor.igamma(other)` 和 `torch.Tensor.igammac(other)` 的 API，以及相应inplace版本。

PyTorch 中输入 CPU 支持 float16, bfloat16, float32, float64，GPU支持 float32, float64

因为 PyTorch 中这些 API 的实际计算逻辑相似性较大，因此下文的分析均以 igammac 为例。

在 PyTorch (aten/src/ATen/native/Math.h)中，不完全伽马函数的核心计算逻辑是 `calc_igammac`/`calc_igammacc` 函数，
这是一个`inline`函数，后续进行了不同的向量化操作，核心计算逻辑代码如下
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

针对一般 float 的向量化处理，主要是给`Vectorized`结构体添加一个新方法，代码如下(aten/src/ATen/cpu/vec/vec256/vec256_float.h)
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

针对 CUDA，核心计算逻辑代码如下(aten/src/ATen/native/cuda/IGammaKernel.cu)，这部分与 CPU的实现相似
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

然后通过一些内部的机制，例如 `AT_DISPATCH_FLOATING_TYPES` 对其处理，代码如下：
```cpp
template<typename scalar_t>
struct CalcIgamma{
  CalcIgamma(bool calc_igammac): calc_igammac_(calc_igammac){}
  bool calc_igammac_;
  __device__ scalar_t operator() (scalar_t a, scalar_t b) const {
    if (calc_igammac_) {
      return calc_igammac(a,b);
    } else {
      return calc_igamma(a,b);
    }
  }
};

void igammac_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_cuda", [&]() {
    gpu_kernel(iter, CalcIgamma<scalar_t>(true));
  });
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
    input: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igammac(
    input: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igamma_(
    input: Tensor,
    other: Tensor,
    name: str | None = None
)
```

```python
paddle.igammac_(
    input: Tensor,
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
Kernel部分CPU实现添加在 `paddle/phi/kernels/cpu/igamma_kernel.cc` 和 `paddle/phi/kernels/cpu/igammac_kernel.cc`，
Kernel部分GPU实现添加在 `paddle/phi/kernels/gpu/igamma_kernel.cu` 和 `paddle/phi/kernels/gpu/igammac_kernel.cu`，
输入 CPU 支持 float16, bfloat16, float32, float64，GPU支持 float32, float64,
对于底层 OP 主要分为三部分，由于 `igamma` 和 `igammac`是互补关系，所以实际上可复用代码很多，
因此底层OP设计仅以`igammac`为例。

### 实现基础计算逻辑
根据 igamma (上不完全伽马函数) 的定义，即
$\Gamma(a, x) = \int_x^{\infty} t^{a-1} e^{-t} dt $
设计相应的CPU和CUDA计算函数（CPU和CUDA主体逻辑相似，仅写法上会存在一些差异），这部分与PyTorch相似，也是最核心的内容。

### 实现基础计算逻辑的向量化（针对CPU）
可采用类似 PyTorch 的向量化技术加速。

### 实现基础计算逻辑的向量化（针对GPU）
这里的 hip 和 cuda 的实现可利用 Paddle 已经实现的很多宏或函数，从而消除两者的差异，
最终实现 Kernel 函数。

## API实现方案

该 API 实现于 `python/paddle/tensor/math.py`。

### igamma
对于 igamma 、 igamma_ 、igammac 和 igammac_ 有类似的API，下面列出了`igamma`的情况。

具体的API为`paddle.igamma(input, other, name = None)`和`paddle.Tensor.igamma(input, other)`

- input: 输入张量，即公式中的 $a$, CPU 支持 float16, bfloat16, float32, float64，GPU支持 float32, float64
- other: 输入张量，即公式中的 $x$, CPU 支持 float16, bfloat16, float32, float64，GPU支持 float32, float64


例如将一维张量 $[3, 5]$ 和一维张量 $[2, 7]$ 输入，则计算结果如下：
$\Gamma(a, x) = [\int_2^{\infty} t^{2} e^{-t} dt, \int_7^{\infty} t^{4} e^{-t} dt]$

# 六、测试和验收的考量
1. 添加单测文件 `test/legacy_test/test_igamma_op.py` 和 `test/legacy_test/test_igamma_op.py`。
2. 在单测文件 `test/legacy_test/test_inplace.py` 补充测试。

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 scipy 作为参考标准
- 对不同 dtype 的输入数据 `input` 和 `other` 进行计算精度检验，与PyTorch保持一致
- 输入输出的容错性与错误提示信息
- 输出 Dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景

# 七、可行性分析和排期规划

方案主要根据相关数学原理并参考 PyTorch 的工程实现方法，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响

# 名词解释
gammainc 是 igamma 的另一种写法，即上不完全伽马函数，gammaincc 是 igammac 的另一种写法，即下不完全伽马函数。

# 附件及参考资料
- [不完全伽马函数的定义——小时百科](https://wuli.wiki/online/IncGam.html)
- [torch.igamma](https://pytorch.org/docs/stable/special.html#torch.special.gammainc)
- [torch.igammac](https://pytorch.org/docs/stable/special.html#torch.special.gammaincc)
- [scipy](https://github.com/scipy/scipy)
- [tensorflow](https://github.com/tensorflow/tensorflow)

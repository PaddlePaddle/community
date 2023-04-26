# paddle.i0 和 paddle.i0e 设计文档
| API 名称     | paddle.i0 / paddle.i0e            |
| ------------ | --------------------------------- |
| 提交作者     | PommesPeter                       |
| 提交时间     | 2023-04-26                        |
| 版本号       | V1.2                              |
| 依赖飞桨版本 | develop                           |
| 文件名       | 20230426_api_design_for_i0&i0e.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算相关 API，Paddle 需要扩充 API `paddle.i0`, `paddle.i0e`。

## 2、功能目标

根据输入的 tensor，计算其每个元素的第一类零阶修正贝塞尔函数（对应 api：`paddle.i0`）和第一类指数缩放的零阶修正贝塞尔函数（对应 api：`paddle.i0e`）。

## 3、意义

为 Paddle 增加计算第一类零阶修正贝塞尔函数和第一类指数缩放的零阶修正贝塞尔函数，丰富 `paddle` 中科学计算相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `i0` 和 `i0e` API，无法计算第一类零阶修正贝塞尔函数和第一类指数缩放零阶修正贝塞尔函数。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `torch.special.i0` 的 API，详细参数为 `torch.special.i0(input, *, out=None) → Tensor`。
PyTorch 中有 `torch.special.i0e` 的 API，详细参数为 `torch.special.i0e(input, *, out=None) → Tensor`。

### i0

> Computes the zeroth order modified Bessel function of the first kind for each element of `input`.

在实现方法上，PyTorch 是通过 C++ API 组合实现的

实现代码：

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/torch/csrc/api/include/torch/special.h#L456-L471)


```cpp
/// Computes the zeroth order modified Bessel function of the first kind of
/// input, elementwise See
/// https://pytorch.org/docs/master/special.html#torch.special.i0
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0(t);
/// ```
inline Tensor i0(const Tensor& self) {
  return torch::special_i0(self);
}

inline Tensor& i0_out(Tensor& result, const Tensor& self) {
  return torch::special_i0_out(result, self);
}
```

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/native/Math.h#L1458-L1474)

```cpp
template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
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

### i0e

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/torch/csrc/api/include/torch/special.h#L490-L505)

```cpp
/// Computes the exponentially scaled zeroth order modified Bessel function of
/// the first kind See
/// https://pytorch.org/docs/master/special.html#torch.special.i0e.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0e(t);
/// ```
inline Tensor i0e(const Tensor& self) {
  return torch::special_i0e(self);
}

inline Tensor& i0e_out(Tensor& result, const Tensor& self) {
  return torch::special_i0e_out(result, self);
}
```

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/native/Math.h#L101-L125)

```cpp
template <typename T>
JITERATOR_HOST_DEVICE T calc_i0e(T _x) {
T x = std::fabs(_x);

if (x <= T{8.0}) {
    static const T coefficients[] = {
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

    T y = (x / T{2.0}) - T{2.0};
    return chbevl(y, coefficients, int{30});
}
```

参数表：

- input (Tensor) – the input tensor
- out (Tensor, optional) – the output tensor.

## Scipy

> Modified Bessel function of order 0.
> Defined as,
> $$I_0(x)=\sum^{\infty}_{k}=0\frac{(x^2/4)^k}{(k!)^2}=J_0(ix)$$
> where $J_0$ is the Bessel function of the first kind of order 0.

在实现方法上，Scipy 是通过 C 代码实现，参考了数据函数库 Cephes 的写法。

```c
/* Chebyshev coefficients for exp(-x) I0(x)
 * in the interval [0,8].
 *
 * lim(x->0){ exp(-x) I0(x) } = 1.
 */
static double A[] = {
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
};

/* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
 * in the inverted interval [8,infinity].
 *
 * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
 */
static double B[] = {
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
};

double i0(x)
double x;
{
    double y;

    if (x < 0)
        x = -x;
    if (x <= 8.0)
    {
        y = (x / 2.0) - 2.0;
        return (exp(x) * chbevl(y, A, 30));
    }
    return (exp(x) * chbevl(32.0 / x - 2.0, B, 25) / sqrt(x));
}

double i0e(x)
double x;
{
    double y;
    if (x < 0)
        x = -x;
    if (x <= 8.0) {
        y = (x / 2.0) - 2.0;
        return (chbevl(y, A, 30));
    }
    return (chbevl(32.0 / x - 2.0, B, 25) / sqrt(x));
}
```

## TensorFlow

### i0

TensorFlow 中已有 `i0` 算子的反向传播实现，可作为参考。

```python
@ops.RegisterGradient("BesselI0")
def _BesselI0Grad(op, grad):
  """Compute gradient of bessel_i0(x) with respect to its argument."""
  x = op.inputs[0]
  with ops.control_dependencies([grad]):
    partial_x = special_math_ops.bessel_i1(x)
    return grad * partial_x
```

### i0e

TensorFlow 中已有 `i0e` 算子的反向传播实现，可作为参考。

```python
@ops.RegisterGradient("BesselI0e")
def _BesselI0eGrad(op, grad):
  """Compute gradient of bessel_i0e(x) with respect to its argument."""
  x = op.inputs[0]
  y = op.outputs[0]
  with ops.control_dependencies([grad]):
    partial_x = (special_math_ops.bessel_i1e(x) - math_ops.sign(x) * y)
    return grad * partial_x
```

# 四、对比分析

## 共同点

- 都能通过输入的 tensor，计算其每个元素的第一类零阶修正贝塞尔函数和第一类指数缩放的零阶修正贝塞尔函数
- 都支持对向量或者矩阵进行运算，对向量或矩阵中的每一个元素计算，是一个 element-wise 的操作
- 都有提供对 Python 的调用接口

## 不同点

- PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。
- Scipy 则是参考数学函数库 Cephes 实现对应的 API

# 五、设计思路与实现方案

## 命名与参数设计

添加 Python API

```python
paddle.i0(
    x: Tensor,
    name: str=None
)
```

```python
paddle.i0e(
    x: Tensor,
    name: str=None
)
```

## 底层OP设计

### i0

参考 Scipy 的 i0 和 i0e 进行实现，实现位置为 Paddle repo 的 `paddle/phi/kernels`。可参考 Cephes 数学函数库的实现方式。

观察公式：

$$out_i=I_0(x)=\sum_{k=0}^{\infty}\frac{(x_i^2/4)^k}{(k!)^2}$$

对于 i0，基于实现的 i0e 乘以缩放系数即可。公式为：

参考 Scipy 的 `i0` 函数进行实现，需要调用 `i0e` 算子，实现位置为 Paddle repo 的 `paddle/phi/kernels`。前向传播计算过程为：

1. 对输入计算绝对值 `y = abs(x)`
2. 对 `y` 计算自然指数 `y' = exp(y)`
3. 调用 `i0e` 得到结果 `out = y' * i0e(y)`

对 `i0(x)` 函数求偏导为 `i1(x)`, 故反向传播计算过程为：

1. 调用 `i1` 计算 `i0` 的偏导数 `partial_x`
2. 计算 `grad * partial_x`

### i0e

对于 i0e，基于实现的 i0 乘以缩放系数即可。公式为：

$$out_i=\exp(-|x|) * I_0(x_i)=\exp(-|x|) * \sum_{k=0}^{\infty}\frac{(x_i^2/4)^k}{(k!)^2}$$

将定义域分为 $[0, 8]$ 和 $[8, \infty]$ 两个区间，在每个区间内部分别通过 Chebyshev 多项式展开计算系数，故提前计算切比雪夫多项式展开系数并代入公式即可。

参考 Scipy 的 `i0e` 函数进行实现，实现位置为 Paddle repo 的 `paddle/phi/kernels`。前向传播计算过程为：

1. 对输入计算绝对值 `y = abs(x)`
2. 如果输入的数值小于 8，则使用 $[0, 8]$ 区间上的 Chebyshev 多项式系数计算 `chbevl((y / 2.0) - 2.0, A)`
3. 如果输入的数值大于 8，则使用 $[8, \infty]$ 区间上的 Chebyshev 多项式系数计算 `chbevl((32.0 / y) - 2.0, B) / sqrt(y)`

对 `i0e(x)` 函数求偏导为 `i1(x) - sign(x) * y`, 故反向传播计算过程为：

1. 调用 `i1` 计算 `i0` 的偏导数 `partial_x`
2. 计算 `grad * partial_x`

## API实现方案

参考 Scipy 进行实现，该 API 实现于 `python/paddle/tensor/math.py` 和 `paddle/phi/kernels`。

# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 scipy 作为参考标准
- 对不同 dtype 的输入数据 `x` 进行计算精度检验 (float32, float64)
- 对不同范围内的输入数据进行计算精度检验 ($[0, 8]$, $[8, \infty]$)
- 输入输出的容错性与错误提示信息
- 输出 Dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景

# 七、可行性分析和排期规划

方案主要依赖现有 Scipy 代码参考实现。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响

# 名词解释

无

# 附件及参考资料

[torch.special.i0](https://pytorch.org/docs/stable/special.html#torch.special.i0)
[torch.special.i0e](https://pytorch.org/docs/stable/special.html#torch.special.i0e)

[scipy.special.i0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0.html#scipy-special-i0)

[scipy.special.i0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html#scipy.special.i0e)

[TensorFlow API 文档](https://www.tensorflow.org/api_docs/python/tf/math/bessel_i1)

[Rational Approximations for the Modified Bessel Function of the First Kind – I1(x) for Computations with Double Precision](https://www.bing.com/search?q=%22Rational+Approximations+for+the+Modified+Bessel+Function+of+the+First+Kind+%0D%0A%23+++++-+I1%28x%29+for+Computations+with+Double+Precision%22+by+Pavel+Holoborodko&qs=n&form=QBRE&sp=-1&lq=1&pq=%22rational+approximations+for+the+modified+bessel+function+of+the+first+kind+%23+-+i1%28x%29+for+computations+with+double+precision%22+by+pavel+holoborodko&sc=0-146&sk=&cvid=2507FDB609B24E39BEA10DE5D3482BD8&ghsh=0&ghacc=0&ghpl=)
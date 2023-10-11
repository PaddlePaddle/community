# paddle.gammaln 设计文档

|API名称 | paddle.gammaln |
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 汪昕([GreatV](https://github.com/GreatV)) |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-25 |
|版本号 | V1.0 |
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop |
|文件名 | 20230925_api_design_for_gammaln.md |

## 一、概述

### 1、相关背景

`gammaln` 函数返回 gamma 函数绝对值的自然对数，定义为：

$$\text{ln}(|\Gamma(x)|)$$

其中 $\Gamma$ 为伽马函数。

### 2、功能目标

为 Paddle 新增 `paddle.gammaln` API，提供 gamma 函数绝对值的自然对数计算功能。

### 3、意义

为 Paddle 新增 `paddle.gammaln` API，提供 gamma 函数绝对值的自然对数计算功能。

## 二、飞桨现状

对飞桨框架目前不支持此功能，且飞桨框架暂无 `paddle.gamma` 函数。但可用其他API组合实现的此功能，代码如下；

```Python
import paddle
import numpy as np


def gammaln(x):
    return paddle.log(paddle.abs(paddle.exp(paddle.lgamma(x))))

np_x = np.array([1/5, 1/2, 2/3, 8/7, 3])
x = paddle.to_tensor(np_x, dtype="float64")

out = gammaln(x)
print(out)
# Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [ 1.52406382,  0.57236494,  0.30315028, -0.06674088,  0.69314718])
```

需要注意的是，此实现在输入x为大值时，会出现数值溢出的情况，如下所示：

```Python
np_x = np.array([1e10, 1e20, 1e40, 1e80])
x = paddle.to_tensor(np_x, dtype="float64")
out = gammaln(x)
print(out)
# Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [inf., inf., inf., inf.])
```

这与我们期望的输出有差异

```Python
# Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [220258509288.81057739                                                               ,
#         4505170185988091674624.                                                             ,
#         911034037197618268983181525026762056007680.                                         ,
#         18320680743952364242533014047349914338837359189968031568576067731446620442113081344.])
```

## 三、业内方案调研

### 1. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.special.gammaln(x, out=None) = <ufunc 'gammaln'>`

其中，`x` 为 `ndarray` 类型, 为实数参数。`out` 为 `ndarray` 类型，是 gamma 函数绝对值的自然对数的输出。Scipy 中的实现与 Python 标准库中的 `math.lgamma` 一致。

scipy 中通过 cephes 库实现，代码如下：

```C
/* Logarithm of Gamma function */
double lgam(double x)
{
    int sign;
    return lgam_sgn(x, &sign);
}

double lgam_sgn(double x, int *sign)
{
    double p, q, u, w, z;
    int i;

    *sign = 1;

    if (!cephes_isfinite(x))
 return x;

    if (x < -34.0) {
 q = -x;
 w = lgam_sgn(q, sign);
 p = floor(q);
 if (p == q) {
   lgsing:
     sf_error("lgam", SF_ERROR_SINGULAR, NULL);
     return (INFINITY);
 }
 i = p;
 if ((i & 1) == 0)
     *sign = -1;
 else
     *sign = 1;
 z = q - p;
 if (z > 0.5) {
     p += 1.0;
     z = p - q;
 }
 z = q * sin(M_PI * z);
 if (z == 0.0)
     goto lgsing;
 /*     z = log(M_PI) - log( z ) - w; */
 z = LOGPI - log(z) - w;
 return (z);
    }

    if (x < 13.0) {
 z = 1.0;
 p = 0.0;
 u = x;
 while (u >= 3.0) {
     p -= 1.0;
     u = x + p;
     z *= u;
 }
 while (u < 2.0) {
     if (u == 0.0)
  goto lgsing;
     z /= u;
     p += 1.0;
     u = x + p;
 }
 if (z < 0.0) {
     *sign = -1;
     z = -z;
 }
 else
     *sign = 1;
 if (u == 2.0)
     return (log(z));
 p -= 2.0;
 x = x + p;
 p = x * polevl(x, B, 5) / p1evl(x, C, 6);
 return (log(z) + p);
    }

    if (x > MAXLGM) {
 return (*sign * INFINITY);
    }

    q = (x - 0.5) * log(x) - x + LS2PI;
    if (x > 1.0e8)
 return (q);

    p = 1.0 / (x * x);
    if (x >= 1000.0)
 q += ((7.9365079365079365079365e-4 * p
        - 2.7777777777777777777778e-3) * p
       + 0.0833333333333333333333) / x;
    else
 q += polevl(p, A, 4) / x;
    return (q);
}
```

### 2. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.special.gammaln(input, *, out=None) → Tensor`

其中，`input` 为 输入`Tensor`，`out` 为 输出 `Tensor`。

Pytorch中直接使用的是C++标准库中的 `std::lgamma`，代码如下：

```C++
  Vectorized<T> lgamma() const {
    return map(std::lgamma);
  }
```

## 四、对比分析

Scipy 基于 cephes 库实现，Pytorch 基于 C++ 标准库实现。Scipy 中的实现与 Python 标准库中的 `math.lgamma` 一致。Pytorch 中直接使用的是C++标准库中的 `std::lgamma`。而飞桨中暂无 `paddle.gamma` 函数，不好通过Python端实现，因此，需要通过C++端实现。

## 五、设计思路与实现方案

### 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.gammaln(x)`。其中，`x` 为 `Tensor` 类型。`Tensor.gammaln()` 为 Tensor 的方法版本。

### API实现方案

参考 PyTorch 采用C++ 实现，实现位置为 Paddle repo `python/paddle/tensor/math.py` 目录。并在 python/paddle/tensor/init.py 中，添加 `gammaln` API，以支持 `Tensor.gammaln` 的调用方式。头文件放在 `paddle/phi/kernels` 目录，cc 文件在 `paddle/phi/kernels/cpu` 目录， cu文件 `paddle/phi/kernels/gpu` 目录。

## 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：

- 输出数值结果的一致性和数据类型是否正确，使用 scipy 作为参考标准；
- 对不同 dtype 的输入数据 `x` 进行计算精度检验 (float32, float64)；
- 大值输入时，是否出现数值溢出；
- 输入输出的容错性与错误提示信息；
- 输出 Dtype 错误或不兼容时抛出异常；
- 保证调用属性时是可以被正常找到的；
- 覆盖静态图和动态图测试场景。

## 七、可行性分析和排期规划

本 API 主要参考 PyTorch 实现，难度适中，工期上能满足要求。

## 八、影响面

为独立新增API，对其他模块没有影响。

## 名词解释

## 附件及参考资料

1. 伽马函数的详细介绍参见 [Gamma Function](https://dlmf.nist.gov/5)

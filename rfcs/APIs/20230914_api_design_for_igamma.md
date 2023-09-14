# paddle.igamma 设计文档

|API名称 | paddle.igamma | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 汪昕([GreatV](https://github.com/GreatV)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-13 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230914_api_design_for_igamma.md |

# 一、概述
## 1、相关背景

`igamma (gammainc)` 用于计算不完全 gamma 函数，下不完全 gamma 函数 $P$ 和上不完全 gamma 函数 $Q$ 由下式定义：

$$P(a, x) = \frac{1}{\Gamma(a)} \int_0^x e^{-t} t^{a - 1} dt$$

$$Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty e^{-t} t^{a - 1} dt$$

其中 `gamma` 函数由下式定义：

$$\Gamma(a)=\int_0^{\infty} e^{-t} t^{a - 1} dt$$

此处使用的 `gamma` 函数的归一化定义，其中 $P(a, x) + Q(a, x) = 1$

不完全 gamma 函数满足如下性质：

- 对于 $a \ge 0$ 有 $\lim\limits_{x \to \infty} P(x, a) = 1$
- $\lim\limits_{x,a \to 0} P(x, a) = 1$

## 2、功能目标

为 Paddle 新增 igamma API。用于计算下不完全 gamma 函数。

## 3、意义

为 Paddle 新增 igamma API，提供不完全 gamma 函数。

# 二、飞桨现状

飞桨框架目前不支持此功能，需要新增 API。

# 三、业内方案调研

## 1. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.special.gammainc(a, x, out=None) = <ufunc 'gammainc'>`

在 Scipy 中，使用 `gammainc` 计算正则化不完全 gamma 函数， 其中 a 为参数 $a > 0$，x 为变量 $x \ge 0$。 

Scipy 的实现主要参考 NIST Digital Library of Mathematical functions [Incomplete Gamma and Related Functions](https://dlmf.nist.gov/8) 和 [Boost](https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html) 的实现。即，计算它们的 series 表示：

$$\gamma(a, x) = x^a e^{-x} \sum_{k=0}^{\infty} \frac{\Gamma(a)}{\Gamma(a + k + 1)} x^n = x^a e^{-x} \sum_{k=0}^{\infty} \frac{x^n}{a^{k \mp 1}}$$

## 2. MatLab

在 MatLab 中使用的 API 格式如下：

`g = igamma(nu,z)`

在 MATLAB 中 ，`igamma` 使用的是上不完全 gamma 函数的定义。然而，默认的 MATLAB `gammainc` 函数使用的是正则化的下不完全 gamma 函数的定义，其中 `gammainc(z,nu)` = `1 - igamma(nu,z)/gamma(nu)` 。

MATLAB 的实现主要参考 NIST Digital Library of Mathematical functions [Incomplete Gamma and Related Functions](https://dlmf.nist.gov/8)。

## 3. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.special.gammainc(input, other, *, out=None) → Tensor`

在 Pytorch 中，使用 `gammainc` 计算正则化下不完全 gamma 函数。

Pytorch 正则化不完全伽马函数及其辅助函数的实现源自 SciPy 的 gammainc、Cephes 的 igam 和 igamc 以及 Boost 的 Lanczos 近似的实现。

## 4. Tensorflow

在 Tensorflow 中使用的 API 格式如下：

`tf.math.igamma(a, x, name=None)`

在 Tensorflow 中，使用 `eigen` 提供的 [igamma](https://eigen.tuxfamily.org/dox/unsupported/namespaceEigen.html#a6e89509c5ff1af076baea462520f231c) 实现此功能。

# 四、对比分析

## 1. 不同框架API使用方式

### 1. Scipy

```Python
import scipy.special as sc
sc.gammainc(0.5, [0, 1, 10, 100])
# [0.         0.84270079 0.99999226 1.        ]
```

### 2. MatLab

```matlab
gammainc([0, 1, 10, 100], 0.5)

// ans = 
//    0 0.8427 1.0000 1.0000
```

### 3. PyTorch

```Python
import torch
​
a1 = torch.tensor([0.5])
a2 = torch.tensor([0.0, 1.0, 10.0, 100.0])
a = torch.special.gammainc(a1, a2)
​
# tensor([0.0000, 0.8427, 1.0000, 1.0000])
```

### 4. Tensorflow

```Python
import tensorflow as tf

a1 = tf.convert_to_tensor([0.5])
a2 = tf.convert_to_tensor([0.0, 1.0, 10.0, 100.0])
a = tf.math.igamma(a1, a2)

# tf.Tensor([0.         0.84270084 0.99999225 1.        ], shape=(4,), dtype=float32)
```

上述框架实现方式类似，可参考 PyTorch 实现。

# 五、设计思路与实现方案

## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.igamma(a, x)`。其中，`a` 为参数 $a > 0$，`x` 为变量 $x \ge 0$。`paddle.gammainc(a, x)` 为 `paddle.igamma(a, x)` 的别名。`igamma_(a, x)` 为 `igamma(a, x)` 的 inplace 版本。`Tensor.igamma(a, x)` 和 `Tensor.igammac(a, x)` 做为 Tensor 的方法使用，`Tensor.igamma_(a, x)` 和 `Tensor.igammac_(a, x)` 做为 Tensor 方法的inplace版本。

## API实现方案

C ++/CUDA 参考 PyTorch 实现，实现位置为 Paddle repo `paddle/phi/kernels` 目录，cc 文件在 `paddle/phi/kernels/cpu` 目录和 cu 文件在 `paddle/phi/kernels/gpu` 目录。Python 实现代码 & 英文 API 文档，放在 Paddle repo 的 `python/paddle/tensor/manipulation.py` 文件。

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：

1. 结果一致性，paddle 实现和 SciPy 以及 PyTorch 结果的数值的一致性；
2. 不同数据类型，对结果精度的影响，如 `dtype=float32` 和 `dtype=float64`；
3. 静态图和动态图均需测试覆盖；
4. 异常测试，对于参数异常值输入，如当 $a < 0, x < 0$ 应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划

本 API 主要参考 PyTorch 实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料

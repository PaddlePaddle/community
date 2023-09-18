# paddle.multigammaln 设计文档

|API名称 | paddle.multigammaln | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 汪昕([GreatV](https://github.com/GreatV)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-13 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230913_api_design_for_multigammaln.md | 


# 一、概述
## 1、相关背景

`multigammaln` 函数返回多元 gamma 函数的对数，有时也称为广义 gamma 函数。对于 $d$ 维实数 $a$ 的多元 gamma 函数定义为：

$$\Gamma_d(a) = \int_{A > 0} {e^{-{tr}(A)}|A|^{a - (d + 1) / 2}} dA $$

其中 $a > (d - 1) / 2$ 且 $A > 0$ 为正定矩阵。上式可写为更加友好的形式：

$$\Gamma_d(a) = \pi^{d(d - 1) / 4} \prod_{i = 1}^d \Gamma(a - (i - 1) / 2)$$

对上式取对数：

$$\log \Gamma_d(a) = \frac{d(d - 1)}{4} \log \pi + \sum_{i = 1}^d \log \Gamma(a - (i - 1) / 2)$$

## 2、功能目标

为 Paddle 新增 `paddle.multigammaln` API，提供多元 gamma 函数的对数计算功能。所有元素必须大于 (d - 1) / 2，否则将会产生未定义行为。

## 3、意义

为 Paddle 新增 `paddle.multigammaln` API，提供多元 gamma 函数的对数计算功能。

# 二、飞桨现状

对飞桨框架目前不支持此功能，可用其他API组合实现的此功能，代码如下；

```Python
import paddle
import numpy as np

a = paddle.to_tensor(23.5)
d = paddle.to_tensor(10)
pi = paddle.to_tensor(np.pi, dtype="float32")

out = (
    d * (d - 1) / 4 * paddle.log(pi)
    + paddle.lgamma(a - 0.5 * paddle.arange(0, d, dtype="float32")).sum()
)

print(out)
```

# 三、业内方案调研

## 1. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.special.multigammaln(a, d)`

其中，`a` 为 `ndarray` 类型，是多元 gamma 函数的变量，`d` 为 `int` 类型，是多元 gamma 函数积分空间的维度。

实现的伪代码如下：

```Python
import numpy as np
from scipy.special import gammaln as loggam


def multigammaln(a, d):
    res = (d * (d - 1) * 0.25) * np.log(np.pi)
    res += np.sum(loggam([(a - (j - 1.0) / 2) for j in range(1, d + 1)]), axis=0)
    return res
```

## 2. jax

在 jax 中使用的 API 格式如下：

`jax.scipy.special.multigammaln(a, d)`

其中，`a` 为 `ndarray` 类型，是多元 gamma 函数的变量，`d` 为 `int` 类型，是多元 gamma 函数积分空间的维度。

实现代码如下：

```python
def multigammaln(a: ArrayLike, d: ArrayLike) -> Array:
  d = core.concrete_or_error(int, d, "d argument of multigammaln")
  a, d_ = promote_args_inexact("multigammaln", a, d)

  constant = lax.mul(lax.mul(lax.mul(_lax_const(a, 0.25), d_),
                             lax.sub(d_, _lax_const(a, 1))),
                     lax.log(_lax_const(a, np.pi)))
  b = lax.div(jnp.arange(d, dtype=d_.dtype), _lax_const(a, 2))
  res = jnp.sum(gammaln(jnp.expand_dims(a, axis=-1) -
                        jnp.expand_dims(b, axis=tuple(range(a.ndim)))),
                axis=-1)
  return res + constant
```
## 3. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.special.multigammaln(input, p, *, out=None)`

其中，`input` 为 `Tensor` 类型，是多元 gamma 函数的变量，`p` 为 `int` 类型，是多元 gamma 函数的积分空间的维度。

实现代码如下：

```python
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType:
    c = 0.25 * p * (p - 1) * math.log(math.pi)
    b = 0.5 * torch.arange(start=(1 - p), end=1, step=1, dtype=a.dtype, device=a.device)
    return torch.sum(torch.lgamma(a.unsqueeze(-1) + b), dim=-1) + c
```


# 四、对比分析

## 1. 不同框架API使用方式

### 1. Scipy

```Python
from scipy.special import multigammaln

a = 23.5
d = 10
out = multigammaln(a, d)
```

### 2. PyTorch

```Python
import torch
​
a = torch.empty(2, 3).uniform_(1, 2)
torch.special.multigammaln(a, 2)
```


上述框架从使用体验来说，差异不大，都是直接调用 API 即可。内部实现上也是大同小异。因此，可参考 PyTorch 的实现，为 Paddle 新增 `paddle.multigammaln` API。

# 五、设计思路与实现方案

## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.multigammaln(x, p)`。其中，`x` 为 `Tensor` 类型，是多元 gamma 函数的变量，`p` 为 `int` 类型，是多元 gamma 函数的积分空间的维度。`paddle.multigammaln_(x, p)` 为 inplace 版本。`Tensor.multigammaln(p)` 为 Tensor 的方法版本。`Tensor.multigammaln_(p)` 为 Tensor 的 方法 inplace 版本。

## API实现方案

参考 PyTorch 采用现有 PYTHON API 组合实现，实现位置为 Paddle repo `python/paddle/tensor/math.py` 目录。并在 python/paddle/tensor/init.py 中，添加 `mvlgamma` & `mvlgamma_` API（alias for `multigammaln`），以支持 `paddle.Tensor.mvlgamma` & `paddle.Tensor.mvlgamma_` 的调用方式

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：

1. 当 `x` 为空张量，输出为空张量，且输出张量形状正确；
2. 结果一致性，和 SciPy 以及 PyTorch 结果的数值的一致性, `paddle.multigammaln(x, p)` , `scipy.special.multigammaln(a, d)` 和 `torch.special.multigammaln(input, p, *, out=None)` 结果是否一致；
3. 异常测试，对于 `x < (p - 1) / 2`，应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划

本 API 主要参考 PyTorch 实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料

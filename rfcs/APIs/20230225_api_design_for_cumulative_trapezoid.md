# paddle.cumulative_trapezoid 设计文档

| API 名称     | cumulative_trapezoid                            |
| ------------ | ----------------------------------------------- |
| 提交作者     | Cattidea                                        |
| 提交时间     | 2023-02-25                                      |
| 版本号       | V1.0                                            |
| 依赖飞桨版本 | develop                                         |
| 文件名       | 20230225_api_design_for_cumulative_trapezoid.md |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.cumulative_trapezoid 和 Tensor.cumulative_trapezoid。

## 2、功能目标

实现  [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)  的算法，支持输入 N 维 Tensor，在指定的某一维实现  [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)  算法。

## 3、意义

为 paddle 框架中提供一种使用 paddle.cumsum 对函数的左右黎曼和求平均来逼近函数定积分的技术(即 trapezoid rule)。

# 二、飞桨现状

飞桨内暂时无相关近似求积分的 API，本次 Hackathon 会贡献一个 paddle.trapezoid 的 API，而 paddle.cumulative_trapezoid 与 paddle.trapezoid 的区别就是在计算加和的时候使用了 cumsum 。

# 三、业内方案调研

## Pytorch

Pytorch 中有 `torch.cumulative_trapezoid(y, x=None, dx=None, dim= -1)` API，沿维度实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)

在 PyTorch 中，介绍为：

> Computes the trapezoidal rule along `dim`. By default the spacing between elements is assumed to be 1, but `dx` can be used to specify a different constant spacing, and `x` can be used to specify arbitrary spacing along `dim`.
> The difference between torch.trapezoid() and this function is that, torch.trapezoid() returns a value for each integration, where as this function returns a cumulative value for every spacing within the integration. This is analogous to how .sum returns a value and .cumsum returns a cumulative sum
>
> Assuming `y` is a one-dimensional tensor with elements $y_0​,y_1​,...,y_n$​, the default computation is
>
> $$
> \sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> When `dx` is specified the computation becomes
>
> $$
> \sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> effectively multiplying the result by `dx`. When `x` is specified, assuming `x` is also a one-dimensional tensor with elements $x_0​,x_1​,...,x_n$​, the computation becomes
>
> $$
> \sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> When `x` and `y` have the same size, the computation is as described above and no broadcasting is needed. The broadcasting behavior of this function is as follows when their sizes are different. For both `x` and `y`, the function computes the difference between consecutive elements along dimension `dim`. This effectively creates two tensors, x_diff and y_diff, that have the same shape as the original tensors except their lengths along the dimension `dim` is reduced by 1. After that, those two tensors are broadcast together to compute final output as part of the trapezoidal rule. See the examples below for details.

官方文档链接为：https://pytorch.org/docs/stable/generated/torch.cumulative_trapezoid.html?highlight=cumulative_trapezoid#torch.cumulative_trapezoid

## TensorFlow && NumPy

TensorFlow 和 NumPy 中均没有 cumulative_trapezoid API 的直接实现，但是有 trapezoid 的相关实现，而 cumulative_trapezoid 和 trapezoid 最大的区别是所用的求和 API 不同。

## SciPy

SciPy 库中有 `scipy.integrate.cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None)` 的实现，使用 trapezoid rule 对 y(x)进行累计积分。

Parameters：

- y(array_like):Values to integrate.
- x(array_like, optional):The coordinate to integrate along. If None (default), use spacing dx between consecutive elements in y.
- dx(int, optional):Specifies the axis to cumulate. Default is -1 (last axis).
- initial:(scalar, optional):If given, insert this value at the beginning of the returned result. Typically this value should be 0. Default is None, which means no value at x[0] is returned and res has one element less than y along the axis of integration.

Reutrns:

- res(ndarray):The result of cumulative integration of y along axis. If initial is None, the shape is such that the axis of integration has one less value than y. If initial is given, the shape is equal to that of y.

官方文档链接：https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html#scipy.integrate.cumulative_trapezoid

# 实现方法

## Pytorch

PyTorch 中实现 trapezoid 使用 C++ [代码实现](https://github.com/pytorch/pytorch/blob/92e03cd583c027a4100a13682cf65771b80569da/aten/src/ATen/native/Integration.cpp#L86)：

```c++
Tensor cumulative_trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    dim = maybe_wrap_dim(dim, y);
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    Tensor x_viewed;
    if (x.dim() == 1) {
        // See trapezoid for implementation notes
        TORCH_CHECK(x.size(0) == y.size(dim), "cumulative_trapezoid: There must be one `x` value for each sample point");
        DimVector new_sizes(y.dim(), 1); // shape = [1] * y.
        new_sizes[dim] = x.size(0); // shape[axis] = d.shape[0]
        x_viewed = x.view(new_sizes);
    } else if (x.dim() < y.dim()) {
        // See trapezoid for implementation notes
        DimVector new_sizes = add_padding_to_shape(x.sizes(), y.dim());
        x_viewed = x.view(new_sizes);
    } else {
        x_viewed = x;
    }
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);
    Tensor dx = x_right - x_left;

    return do_cumulative_trapezoid(y, dx, dim);
}

Tensor cumulative_trapezoid(const Tensor& y, const Scalar& dx, int64_t dim) {
    TORCH_CHECK(y.scalar_type() != kBool, "cumulative_trapezoid: received a bool input for `y`, but bool is not supported")
    TORCH_CHECK(!(dx.isComplex() || dx.isBoolean()), "cumulative_trapezoid: Currently, we only support dx as a real number.");

    return do_cumulative_trapezoid(y, dx.toDouble(), dim);
}

Tensor do_cumulative_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return ((left + right) * dx).cumsum(dim) / 2.;
}

Tensor do_cumulative_trapezoid(const Tensor& y, double dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);

    return (dx /2. * (left + right)).cumsum(dim);
}
```

## TensorFlow && NumPy

TensorFlow 和 NumPy 中没有对 cumulative_trapezoid API 的直接实现，NumPy 可利用 SciPy 的 scipy.integrate.cumulative_trapezoid API 来对 NumPy 的数据进行计算，TensorFlow 中可以用组合 API 的形式实现。

## SciPy

SciPy 中的 scipy.integrate.cumulative_trapezoid 是通过 [python 代码](https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_quadrature.py#L395-L485)实现的，具体如下：

```Python
def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.
    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.
    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.
    See Also
    --------
    numpy.cumsum, numpy.cumprod
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    ode : ODE integrators
    odeint : ODE integrators
    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()
    """
    y = np.asarray(y)
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    return res

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)
```

# 四、对比分析

**主要分析 cumulative_trapezoid API 和 trapezoid API 在具体实现上的差异：**

PyTorch 中的有 `torch.trapezoid(y, x=None, dx=None, dim= -1)` 和 `torch.cumulative_trapezoid(y, x=None, dx=None, dim= -1)` API 的实现。

- 从功能上分析：两者都是在指定维度上使用 trapezoid rule 算法，区别是：trapezoid 使用 sum 函数求和，cumulative_trapezoid 使用 cumsum 函数求和。
- 从参数上分析：两个 API 的参数表对应相同。
- 从代码实现上分析：两个 API 最大的区别在于计算加和的 API 不一样：trapezoid 使用的是 sum API，而 cumulative_trapezoid 使用的是 cumsum API。除此之外，其余逻辑一致。

[PyTorch trapezoid 代码实现](https://github.com/pytorch/pytorch/blob/92e03cd583c027a4100a13682cf65771b80569da/aten/src/ATen/native/Integration.cpp#L86)

[PyTorch cumulative_trapezoid 代码实现](https://github.com/pytorch/pytorch/blob/92e03cd583c027a4100a13682cf65771b80569da/aten/src/ATen/native/Integration.cpp#L86)

NumPy 和 Scpiy 中分别有 `numpy.trapz(y, x=None, dx=1.0, axis=-1)` 和`scipy.integrate.cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None)` API 的实现。

- 从功能上分析：两者都是在指定维度上使用 trapezoid rule 算法，区别是 trapez API 使用 sum 函数求和，cumulative_trapezoid API 使用 cumsum 函数求和。此外，scipy.integrate.cumulative_trapezoid 可以实现结果前插值的功能。
- 从参数上分析：Scpiy 中的 cumulative_trapezoid 的参数多了 `initial`，在返回结果的指定维度前插值，通常是 0，默认值是 None，即表示不进行插值操作。
- 从代码实现上分析：NumPy 中 trapz 和 SciPy 中 cumulative_trapezoid 两点较大的区别在于使用的求和 API 不同，前者是 sum API，后者是 cumsum API；cumulative API 相比于 trapz API 多了一个结果插值的功能。而其余逻辑实现基本保持一致。

[NumPy trapezoid 代码实现](https://github.com/numpy/numpy/blob/8cec82012694571156e8d7696307c848a7603b4e/numpy/lib/function_base.py#L4773-L4884)

[SciPy cumulative_trapezoid 代码实现](https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_quadrature.py#L395-L485)

# 五、设计思路与实现方案

**paddle.cumulative_trapezoid 的设计思路，主要还是参考 TensorFlow 中 trapezoid API 的设计，其中把求和函数改为 cumsum 即可。**

## 命名与参数设计

`paddle.cumulative_trapezoid(y, x=None, dx=None, axis=-1)` 参数说明如下：

- **y** (Tensor) – 要积分的张量。
- **x** (Tensor) – 可选，**y** 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。
- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为 1.0。
- **axis** (int) – 计算  trapezoidal rule 时 **y** 的维度。默认值-1。

`Tensor.cumulative_trapezoid(x=None, dx=None, axis=-1)` 参数说明如下：

- **x** (Tensor) – 可选，`Tensor` 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。
- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为 1.0。
- **axis** (int) – 计算  trapezoidal rule 时 **y** 的维度。

输出是一个 Tensor，其形状与 **y** 的形状与用于计算  trapezoidal rule 时的维度有关。

## 底层 OP 设计

主要使用 paddle.diff、paddle.cumsum 等现有 API 进行设计。

## API 实现方案

- 确保 `x` 和 `dx` 不都为空
- 确保输入 `axis` 是一个有效的 `int` 维度
- 若 `x` 为空
  - 若 `dx` 为空
    - 设 `dx` 等于 1，根据下式按照指定维度直接计算结果：
      $$\sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)$$
  - 若 `dx` 不为空，根据下式按照指定维度直接计算结果：
    $$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$
- 若 `x` 不为空：按照指定维度利用 paddle.diff 差分求解间距，根据下式直接计算结果：
  $$\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)$$

# 六、测试和验收的考量

(与[paddle 新增 paddle.trapezoid API](https://github.com/PaddlePaddle/community/pull/373)中验收标准保持一致)

1. 结果正确性:

   - 前向计算: `paddle.cumulative_trapezoid` 和 `Tensor.cumulative_trapezoid` 计算结果与 `scipy.integrate.cumulative_trapezoid` 计算结果一致。
   - 反向计算:由 Python 组合新增 API 无需验证反向计算。

2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

3. 异常测试:

   - 数据类型检验:
     - y 要求为 paddle.Tensor
     - x 若有输入, 则要求为 paddle.Tensor
     - dx 若有输入, 则要求为 float
     - axis 若有输入, 则要求为 int
   - 具体数值检验:
     - 若 x 有输入, 已知 y 的尺寸为 `[d_1, d_2, ... , d_n]` 且 `axis=k` , 则 x 的尺寸只能为 `[d_k]` 或 `[d_1, d_2, ... , d_n]`
     - 若 dx 有输入, 则要非负
     - 若 axis 有输入, 则要求 y 存在该维度

4. 各参数输入组合有效:
   - 检查只输入 y 的情况
     - 正常计算
   - 检查输入 y 和 dx 的情况
     - 正常计算
   - 检查输入 y 和 x 的情况
     - 正常计算
   - 检查输入 y, dx 和 axis 的情况
     - 检查 y 是否存在输入的 axis 索引, 若存在则正常计算; 否则抛出异常
   - 检查输入 y, x 和 axis 的情况
     - 检查 y 是否存在输入的 axis 索引, 若存在则正常计算; 否则抛出异常
     - 检查 x 和 y 的尺寸是否匹配, 若存在则正常计算; 否则抛出异常
     - 其余情况正常计算
   - 其他组合输入
     - 异常组合输入, 抛出异常

# 七、可行性分析和排期规划

方案主要依赖现有 paddle API 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[[Hackathon 4th No.3]为 paddle 新增 paddle.trapezoid API](https://github.com/PaddlePaddle/community/pull/373)  
[PyTorch 官方文档](https://pytorch.org/docs/stable/generated/torch.cumulative_trapezoid.html?highlight=cumulative_trapezoid#torch.cumulative_trapezoid)  
[SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html#scipy.integrate.cumulative_trapezoid)

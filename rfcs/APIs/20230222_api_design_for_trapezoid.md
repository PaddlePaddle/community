# paddle.trapezoid 设计文档

| API 名称     | trapezoid                            |
| ------------ | ------------------------------------ |
| 提交作者     | Cattidea                             |
| 提交时间     | 2023-02-22                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop                              |
| 文件名       | 20230222_api_design_for_trapezoid.md |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.trapezoid 和 Tensor.trapezoid。

## 2、功能目标

实现  [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)  的算法，支持输入 N 维 Tensor，在指定的某一维实现  [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)  算法。

## 3、意义

为 paddle 框架中提供一种通过对函数的左右黎曼和求平均来逼近函数定积分的技术(即 trapezoid rule)。

# 二、飞桨现状

飞桨内暂时无相关近似求积分的 API。

# 三、业内方案调研

## Pytorch

Pytorch 中有 `torch.trapezoid(y, x=None, dx=None, dim= -1)` API，沿维度实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)

在 PyTorch 中，介绍为：

> Computes the trapezoidal rule along `dim`. By default the spacing between elements is assumed to be 1, but `dx` can be used to specify a different constant spacing, and `x` can be used to specify arbitrary spacing along `dim`.
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
> 官方文档链接为：https://pytorch.org/docs/stable/generated/torch.trapezoid.html?highlight=trapezoid#torch.trapezoid

- 如果输入的`dx`是 double，即相邻两个采样点间隔相同，则计算如下：
  $$\sum*{i=1}^{n-1}\frac{\Delta x}{2}\left(y*{i}+y\_{i-1}\right)$$
- 如果输入的`dx`是 tensor，即使用 tensor 指定相邻两个采样点间隔，则计算如下 (其中 $\Delta x_i = x_{i+1}-x_i$）：
  $$\sum_{i=1}^{n-1} \frac{\Delta x_i}{2}\left(y_{i}+y_{i-1}\right)$$

官方文档链接为：https://pytorch.org/docs/stable/generated/torch.trapezoid.html?highlight=trapezoid#torch.trapezoid

## Tensorflow

tensorflow 中有 API `tfp.math.trapz(y, x=None, dx=None, axis=-1, name=None)`，在指定轴上实现 trapezoidal relu 算法，公式如下：
$$\int y(x) dx \approx \sum_{k=0}^n 0.5\times(y_k+y{k+1})\times(x_{k+1}-x_k)$$
**Args**:

- **y**:Float Tensor of values to integrate.
- **x**:Optional, Float Tensor of points corresponding to the y values. The shape of x should match that of y. If x is None, the sample points are assumed to be evenly spaced dx apart.
- **dx**:Scalar float Tensor. The spacing between sample points when x is None. If neither x nor dx is provided then the default is dx = 1.
- **axis**: Scalar integer Tensor. The axis along which to integrate.
- **name**: Python str name prefixed to ops created by this function. Default value: None, uses name='trapz'.

**Return**:Float Tensor integral approximated by trapezoidal rule. Has the shape of y but with the dimension associated with axis removed.
官方文档链接为：https://www.tensorflow.org/probability/api_docs/python/tfp/math/trapz

## Numpy

Numpy 中有 API `numpy.trapz(y, x=None, dx=1.0, axis=-1)` 使用 trapezoidal relu 沿特定轴进行计算，计算公式如下：

$$
\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)
$$

如果参数 `x` 有效，则沿着 `x` 中元素的顺序进行计算。沿着给定轴对 y(x)积分，计算$\int y(x) dx$ 。当指定`x`时，计算$\int_ty(t)d_t=\int_ty(t) {d_x \over d_t}|_{x=x(t)}d_t$。
参数：

- y(array-like)：要积分的数组。
- x(array-like, optional)：y 值对应的样本点。如果 x 为 None，则假定样本点均匀间隔 dx，默认值为 None。
- dx(scalar, optional)：当 x 为 None 是，采样点的默认间距，默认值为 1。
- axis(int, optional)：指定的积分的轴，默认值为-1。
  Returns:
- trapz(float or ndarray)：按 trapezoidal relu 对 y 积分得到的数组。如果 y 是一维数组，则结果为浮点数，如果 y 的维数大于 1，假设 n 维，则结果为 n-1 维的数组。

官方文档链接为：https://numpy.org/doc/stable/reference/generated/numpy.trapz.html#numpy.trapz

# 实现方法

## Pytorch

PyTorch 中实现 trapezoid 使用 C++ [代码实现](https://github.com/pytorch/pytorch/blob/92e03cd583c027a4100a13682cf65771b80569da/aten/src/ATen/native/Integration.cpp#L86)：

```c++
Tensor trapezoid(const Tensor& y, const Tensor& x, int64_t dim) {
    dim = maybe_wrap_dim(dim, y);
    // asking for the integral with zero samples is a bit nonsensical,
    // but we'll return "0" to match numpy behavior.
    if (y.size(dim) == 0) {
        return zeros_like_except(y, dim);
    }
    TORCH_CHECK(y.scalar_type() != kBool && x.scalar_type() != kBool, "trapezoid: received a bool input for `x` or `y`, but bool is not supported")
    Tensor x_viewed;
    // Note that we explicitly choose not to broadcast 'x' to match the shape of 'y' here because
    // we want to follow NumPy's behavior of broadcasting 'dx' and 'dy' together after the differences are taken.
    if (x.dim() == 1) {
        // This step takes 'x' with dimension (n,), and returns 'x_view' with
        // dimension (1,1,...,n,...,1,1) based on dim and y.dim() so that, later on, 'dx'
        // can be broadcast to match 'dy' at the correct dimensions.
        TORCH_CHECK(x.size(0) == y.size(dim), "trapezoid: There must be one `x` value for each sample point");
        DimVector new_sizes(y.dim(), 1); // shape = [1] * y.
        new_sizes[dim] = x.size(0); // shape[axis] = d.shape[0]
        x_viewed = x.view(new_sizes);
    } else if (x.dim() < y.dim()) {
        // When 'y' has more dimension than 'x', this step takes 'x' with dimension (n_1, n_2, ...),
        // and add '1's as dimensions in front to become (1, 1, ..., n_1, n_2), matching the dimension of 'y'.
        // This allows the subsequent slicing operations to proceed with any 'dim' without going out of bound.
        DimVector new_sizes = add_padding_to_shape(x.sizes(), y.dim());
        x_viewed = x.view(new_sizes);
    } else {
        x_viewed = x;
    }
    // Note the .slice operation reduces the dimension along 'dim' by 1,
    // while the sizes of other dimensions are untouched.
    Tensor x_left = x_viewed.slice(dim, 0, -1);
    Tensor x_right = x_viewed.slice(dim, 1);

    Tensor dx = x_right - x_left;
    return do_trapezoid(y, dx, dim);

// The estimated integral of a function y of x,
// sampled at points (y_1, ..., y_n) that are separated by distance (dx_1, ..., dx_{n-1}),
// is given by the trapezoid rule:
//
// \sum_{i=1}^{n-1}  dx_i * (y_i + y_{i+1}) / 2
//
// TODO: if we extend TensorIterator to accept 3 inputs,
// we can probably make this a bit more performant.
Tensor do_trapezoid(const Tensor& y, const Tensor& dx, int64_t dim) {
    Tensor left = y.slice(dim, 0, -1);
    Tensor right = y.slice(dim, 1);
    // If the dimensions of 'dx' and '(left + right)' do not match
    // broadcasting is attempted here.
    return ((left + right) * dx).sum(dim) / 2.;
}
// When dx is constant, the above formula simplifies
// to dx * [(\sum_{i=1}^n y_i) - (y_1 + y_n)/2]
Tensor do_trapezoid(const Tensor& y, double dx, int64_t dim) {
    return (y.sum(dim) - (y.select(dim, 0) + y.select(dim, -1)) * (0.5)) * dx;
}
```

## Tensorflow

Tensorflow 中实现 tfp.math.trapz 使用 python 实现，[实现代码](https://github.com/tensorflow/probability/blob/5639d0e49a4adcf9b3c63279739d500151c80cff/tensorflow_probability/python/math/integration.py#L30)如下：

```Python
def trapz(
    y,
    x=None,
    dx=None,
    axis=-1,
    name=None,
):
  """Integrate y(x) on the specified axis using the trapezoidal rule.
  Computes ∫ y(x) dx ≈ Σ [0.5 (y_k + y_{k+1}) * (x_{k+1} - x_k)]
  Args:
    y: Float `Tensor` of values to integrate.
    x: Optional, Float `Tensor` of points corresponding to the y values. The
      shape of x should match that of y. If x is None, the sample points are
      assumed to be evenly spaced dx apart.
    dx: Scalar float `Tensor`. The spacing between sample points when x is None.
      If neither x nor dx is provided then the default is dx = 1.
    axis: Scalar integer `Tensor`. The axis along which to integrate.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None`, uses name='trapz'.
  Returns:
    Float `Tensor` integral approximated by trapezoidal rule.
      Has the shape of y but with the dimension associated with axis removed.
  """
  with tf.name_scope(name or 'trapz'):
    if not (x is None or dx is None):
      raise ValueError('Not permitted to specify both x and dx input args.')
    dtype = dtype_util.common_dtype([y, x, dx], dtype_hint=tf.float32)
    axis = ps.convert_to_shape_tensor(axis, dtype=tf.int32, name='axis')
    axis_rank = tensorshape_util.rank(axis.shape)
    if axis_rank is None:
      raise ValueError('Require axis to have a static shape.')
    if axis_rank:
      raise ValueError(
          'Only permitted to specify one axis, got axis={}'.format(axis))
    y = tf.convert_to_tensor(y, dtype=dtype, name='y')
    y_shape = ps.convert_to_shape_tensor(ps.shape(y), dtype=tf.int32)
    length = y_shape[axis]
    if x is None:
      if dx is None:
        dx = 1.
      dx = tf.convert_to_tensor(dx, dtype=dtype, name='dx')
      if ps.shape(dx):
        raise ValueError('Expected dx to be a scalar, got dx={}'.format(dx))
      elem_sum = tf.reduce_sum(y, axis=axis)
      elem_sum -= 0.5 * tf.reduce_sum(
          tf.gather(y, [0, length - 1], axis=axis),
          axis=axis)  # half weight endpoints
      return elem_sum * dx
    else:
      x = tf.convert_to_tensor(x, dtype=dtype, name='x')
      tensorshape_util.assert_is_compatible_with(x.shape, y.shape)
      dx = (
          tf.gather(x, ps.range(1, length), axis=axis) -
          tf.gather(x, ps.range(0, length - 1), axis=axis))
      return 0.5 * tf.reduce_sum(
          (tf.gather(y, ps.range(1, length), axis=axis) +
           tf.gather(y, ps.range(0, length - 1), axis=axis)) * dx,
          axis=axis)
```

## Numpy

Numpy 中的 trapz 通过 Python 实现，[具体代码](https://github.com/numpy/numpy/blob/8cec82012694571156e8d7696307c848a7603b4e/numpy/lib/function_base.py#L4773-L4884)如下:

```Python
def trapz(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.
    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.
    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.
    Returns
    -------
    trapz : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.
    See Also
    --------
    sum, cumsum
    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.
    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule
    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png
    Examples
    --------
    >>> np.trapz([1,2,3])
    4.0
    >>> np.trapz([1,2,3], x=[4,6,8])
    8.0
    >>> np.trapz([1,2,3], dx=2)
    8.0
    Using a decreasing `x` corresponds to integrating in reverse:
    >>> np.trapz([1,2,3], x=[8,6,4])
    -8.0
    More generally `x` is used to integrate along a parametric curve.
    This finds the area of a circle, noting we repeat the sample which closes
    the curve:
    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapz(np.cos(theta), x=np.sin(theta))
    3.141571941375841
    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapz(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapz(a, axis=1)
    array([2.,  8.])
    """
    y = asanyarray(y)
    if x is None:
        d = dx
    else:
        x = asanyarray(x)
        if x.ndim == 1:
            d = diff(x)
            # reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    return ret
```

# 四、对比分析

计算思路基本一致，无功能差别，只是在分片方式上不一样。

在计算差分过程中，PyTorch 使用 slice 然后相减，从而完成差分；Tensorflow 使用 gather 然后相减，从而完成差分。Numpy 使用 diff 做差分。paddle.trapezoid API 的设计主要参考 Tensorflow 中的实现，具体逻辑如下：
Tensorflow 实现：

- 首先确保 `x` 和 `dx` 不都为空
- 确保输入 `axis` 是一个有效的 `int` 维度
- 若 `x` 为空
  - 若 `dx` 为空
    - 设 `dx` 等于 1，根据下式按照指定维度直接计算结果：
      $$\sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)$$
  - 若 `dx` 不为空，根据下式按照指定维度直接计算结果：
    $$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$
- 若 `x` 不为空：按照指定维度利用差分求解间距，根据下式直接计算结果：
  $$\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)$$

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.trapezoid(y, x=None, dx=None, axis=-1)` 参数说明如下：

- **y** (Tensor) – 要积分的张量。

- **x** (Tensor) – 可选，**y** 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。

- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为 1.0。

- **axis** (int) – 计算  trapezoidal rule 时 **y** 的维度。默认值-1。

`Tensor.trapezoid(x=None, dx=None, axis=-1)` 参数说明如下：

- **x** (Tensor) – 可选，`Tensor` 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。

- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为 1.0。

- **dim** (int) – 计算  trapezoidal rule 时 **y** 的维度。

输出是一个 Tensor，其形状与 **y** 的形状与用于计算  trapezoidal rule 时的维度有关。

## 底层 OP 设计

主要使用 paddle.diff、paddle.sum 等现有 API 进行设计。

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

1. 结果正确性:

   - 前向计算: `paddle.trapezoid`(和 `Tensor.trapezoid`) 计算结果与 `np.trapz` 计算结果一致。
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

方案主要依赖现有 paddle api 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[【Hackathon No.6】为 Paddle 新增 trapezoid](https://github.com/PaddlePaddle/community/pull/173)  
[Pytorch 官方文档](https://pytorch.org/docs/stable/generated/torch.trapezoid.html?highlight=trapezoid#torch.trapezoid)  
[Tensorflow 官方文档](https://www.tensorflow.org/probability/api_docs/python/tfp/math/trapz)  
[Numpy 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.trapz.html#numpy.trapz)

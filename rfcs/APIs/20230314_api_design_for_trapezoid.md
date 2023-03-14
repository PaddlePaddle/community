# paddle.trapezoid 设计文档

| API名称      | trapezoid                            |
| ------------ |--------------------------------------|
| 提交作者     | [cos43 ](https://github.com/cos43)   |
| 提交时间     | 2023-03-13                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop                              |
| 文件名       | 20230314_api_design_for_trapezoid.md |

# 一、概述

## 1、相关背景

Paddle需要扩充API：paddle.trapezoid 和 Tensor.trapezoid。

## 2、功能目标

实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 的算法，支持输入N 维 Tensor，在指定的某一维实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 算法。

## 3、意义

为 paddle 框架中提供一种通过对函数的左右黎曼和求平均来逼近函数定积分的技术(即trapezoid rule)。

# 二、飞桨现状

飞桨内暂时无相关近似求积分都API。

# 三、业内方案调研

#### Pytorch

Pytorch 中有 API:

`torch.trapezoid(y, x=None, dx=None, dim=- 1)`

在 PyTorch 中，介绍为：

> Computes the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) along `dim`. By default the spacing between elements is assumed to be 1, but `dx` can be used to specify a different constant spacing, and `x` can be used to specify arbitrary spacing along `dim`.
>
> Assuming `y` is a one-dimensional tensor with elements $y_0,y_1,...,y_n$, the default computation is
>
> $$
> \sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> When `dx` is specified the computation becomes
>
> $$
> \sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> effectively multiplying the result by `dx`. When `x` is specified, assuming `x` is also a one-dimensional tensor with elements $x_0,x_1,...,x_n$, the computation becomes
>
> $$
> \sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)
> $$
>
> When `x` and `y` have the same size, the computation is as described above and no broadcasting is needed. The broadcasting behavior of this function is as follows when their sizes are different. For both `x` and `y`, the function computes the difference between consecutive elements along dimension `dim`. This effectively creates two tensors, x_diff and y_diff, that have the same shape as the original tensors except their lengths along the dimension `dim` is reduced by 1. After that, those two tensors are broadcast together to compute final output as part of the trapezoidal rule. See the examples below for details.

C++实现代码：

```cpp
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

- 如果输入的`dx`是double，即相邻两个采样点间隔相同，则计算如下：$$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$
- 如果输入的`dx`是tensor，即使用tensor指定相邻两个采样点间隔，则计算如下 (其中$\Delta x_i = x_{i+1}-x_i$)：$$\sum_{i=1}^{n-1} \frac{\Delta x_i}{2}\left(y_{i}+y_{i-1}\right)$$

#### Tensorflow

tensorflow 中有 API:

`tfp.math.trapz(y, x=None, dx=None, axis=-1, name=None)`

```python
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

整体逻辑为：

- 确保 `x` 和 `dx` 不都为空
- 确保输入 `axis` 是一个有效的 `int` 维度
- 若 `x` 为空
  - 若 `dx` 为空
    - 设 `dx` 等于 1，根据下式按照指定维度直接计算结果：$$\sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)$$
  - 若 `dx` 不为空，根据下式按照指定维度直接计算结果：$$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$
- 若 `x` 不为空：按照指定维度利用差分求解间距，根据下式直接计算结果：$$\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)$$

#### Numpy

Numpy 中有 API:

`numpy.trapz(y, x=None, dx=1.0, axis=-1)`

```python
def trapz(y, x=None, dx=1.0, axis=-1):
    """
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
        Definite integral of 'y' = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If 'y' is a 1-dimensional array,
        then the result is a float. If 'n' is greater than 1, then the result
        is an 'n-1' dimensional array.
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

#### PyTorch

使用`torch.trapezoid(y, x=None, dx=None, dim=- 1)`计算 trapezoidal rule。其中，如果输入的`dx`是double，则计算如下：

$$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$

如果输入的`dx`是tensor，则使用tensor指定相邻两个采样点间隔，计算如下（其中$\Delta x_i = x_{i+1}-x_i$）：

$$\sum_{i=1}^{n-1} \frac{\Delta x_i}{2}\left(y_{i}+y_{i-1}\right)$$

#### Tensorflow

使用`tfp.math.trapz(y, x=None, dx=None, axis=-1, name=None)`计算 trapezoidal rule。其中，

- 确保 `x` 和 `dx` 不都为空
- 确保输入 `axis` 是一个有效的 `int` 维度
- 若 `x` 为空
  - 若 `dx` 为空
    - 设 `dx` 等于 1，根据下式按照指定维度直接计算结果：$$\sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)$$
  - 若 `dx` 不为空，根据下式按照指定维度直接计算结果：$$\sum_{i=1}^{n-1} \frac{\Delta x}{2}\left(y_{i}+y_{i-1}\right)$$
- 若 `x` 不为空：按照指定维度利用差分求解间距，根据下式直接计算结果：$$\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)$$

#### Numpy

使用`numpy.trapz(y, x=None, dx=1.0, axis=-1)`计算 trapezoidal rule。其中，

- 若输入的 `x` 不为空，则按照指定维度计算结果：$$\sum_{i=1}^{n-1} \frac{\left(x_{i}-x_{i-1}\right)}{2}\left(y_{i}+y_{i-1}\right)$$
- 若输入的 `x` 为空，则按照指定维度计算结果：$$\sum_{i=1}^{n-1} \frac{1}{2}\left(y_{i}+y_{i-1}\right)$$

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.trapezoid(y, x=None, dx=None, axis=-1)` 参数说明如下：

- **y** (Tensor) – 计算 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 时所需的值。
  
- **x** (Tensor) – 可选，**y** 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。
  
- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为-1。
  
- **axis** (int) – 计算 trapezoidal rule 时 **y** 的维度。

输出是一个Tensor，其形状与 **y** 的形状与用于计算 trapezoidal rule 时的维度有关。

## 底层OP设计

使用已有API组合实现，不再单独设计OP。

## API实现方案

1. 首先确保 `x` 和 `dx` 不都为空
2. 确保输入 `axis` 是一个有效的 `int` 维度
3. 如果 `x` 是 `None`，则使用 `dx` 作为步长
4. 如果 `x` 是一维 `Tensor`，则使用 `paddle.diff(x) ` 计算步长，并根据 axis 调整形状
5. 如果 `x` 是多维 `Tensor`，则使用 ` paddle.diff(x, axis=axis) `计算步长
6. 使用` (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)` 计算积分值


# 六、测试和验收的考量

测试考量的角度如下:

1. 计算结果与Numpy使用`trapz`函数的计算结果一致。
  
2. CPU、GPU下计算一致。
  
3. 各参数输入有效。


测试样例的构造思路：

- 测试边界值：针对输入参数的边界值进行测试，如 dx 为 0 或负数时，axis 为 -1 或超出范围时等
- 测试覆盖率：尽可能覆盖所有可能的输入情况，包括正常情况和异常情况
- 测试精度：检查输出结果是否与预期结果相符或足够接近

# 七、可行性分析和排期规划

方案主要依赖现有paddle api组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
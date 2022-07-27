# paddle.trapezoid 设计文档

| API名称 | trapezoid |
| --- | --- |
| 提交作者<input type="checkbox" class="rowselector hidden"> | [kunkun0w0](https://github.com/kunkun0w0) |
| 提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-08 |
| 版本号 | V1.0 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop |
| 文件名 | 20220708_api_design_for_trapezoid.md<br> |

# 一、概述

## 1、相关背景

Paddle需要扩充API：paddle.trapezoid 和 Tensor.trapezoid。

## 2、功能目标

实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 的算法，支持输入N 维 Tensor，在指定的某一维实现 [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) 算法。

## 3、意义

为 paddle 框架中提供一种通过对函数的左右黎曼和求平均来逼近函数定积分的技术(即trapezoid rule)。

# 二、飞桨现状

飞桨内暂时无相关近似求积分的API。

# 三、业内方案调研

Pytorch 中有相关的函数

```python
torch.trapezoid(y, x=None, dx=None, dim=- 1)
```

在 PyTorch 中，介绍为：

> Computes the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) along `dim`. By default the spacing between elements is assumed to be 1, but `dx` can be used to specify a different constant spacing, and `x` can be used to specify arbitrary spacing along `dim`.
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

PyTorch C++ 代码：

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

Tensorflow python 代码

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


# 四、对比分析

计算思路基本一致，无功能差别，只是在分片方式上不一样。

在计算差分过程中，PyTorch使用slice然后相减，从而完成差分；Tensorflow使用gather然后相减，从而完成差分。

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.trapezoid(y, x=None, dx=None, axis=-1)` 参数说明如下：

- **y** (Tensor) – 计算 trapezoidal rule 时所需的值。
  
- **x** (Tensor) – 可选，**y** 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。
  
- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为1.0。
  
- **dim** (int) – 计算 trapezoidal rule 时 **y** 的维度。
  

`Tensor.trapezoid(x=None, dx=None, axis=-1)` 参数说明如下：

- **x** (Tensor) – 可选，`Tensor` 中数值对应的点的浮点数所组成的 Tensor；**x** 的形状应与 **y** 的形状相匹配；如果 **x** 为 None，则假定采样点均匀分布 **dx**。
  
- **dx** (float) - 相邻采样点之间的常数间隔；当**x**和**dx**均未指定时，**dx**默认为1.0。
  
- **dim** (int) – 计算 trapezoidal rule 时 **y** 的维度。
  

输出是一个Tensor，其形状与 **y** 的形状与用于计算 trapezoidal rule 时的维度有关。

## 底层OP设计

使用`paddle.diff`和`Tensor.sum`组合实现。

## API实现方案

核心计算公式如下: $$\sum_{i=1}^{n-1} \frac{\Delta x_i}{2}\left(y_{i}+y_{i-1}\right)$$

- 若 `x` 为空
  - 若 `dx` 为空, 则 $\Delta x_i = 1.0$
  - 若 `dx` 不为空, 则 $\Delta x_i = \text{dx}$
- 若 `x` 不为空：按照指定维度进行如下差分：$\Delta x_i = x_{i+1} - x_{i}$


demo:

```python
def trapezoid(y, x=None, dx=1.0, axis=-1):
    if x is None:
        d = dx
    else:
        d = paddle.diff(x, axis=axis)

    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    result = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)

    return result
```

# 六、测试和验收的考量

1. 结果正确性: 
    - 前向计算: `paddle.trapezoid`(和 `Tensor.trapezoid`) 计算结果与 `np.trapz` 计算结果一致。
    - 反向计算: `paddle.trapezoid`(和 `Tensor.trapezoid`) 计算结果反向传播所得到的梯度与使用 numpy 手动计算的结果一致。令输出 $p$ 对 $x_i$ 求导所得梯度为 $g_i$ 则:
        - 当 $i=1$ 时, $g_i = \Delta x_1$ 
        - 当 $\text{1} < \text{i}< \text{n-1}$ 时, $g_i = \frac{\Delta x_{i-1} + \Delta x_{i}}{2}$
        - 当 $i=n$ 时, $g_i = \Delta x_{n-1}$
        
2. 硬件场景: 在CPU和GPU硬件条件下的运行结果一致。
  
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
        
3. 各参数输入组合有效:
    - 检查只输入 y 的情况
        - 正常计算
    - 检查输入 y 和 dx 的情况
        - 正常计算
    - 检查输入 y 和 x 的情况
        - 正常计算
    - 检查输入 y, dx 和 axis 的情况
        - 检查y是否存在输入的axis索引, 若存在则正常计算; 否则抛出异常
    - 检查输入 y, x 和 axis 的情况
        - 检查 y 是否存在输入的 axis 索引, 若存在则正常计算; 否则抛出异常
        - 检查 x 和 y 的尺寸是否匹配, 若存在则正常计算; 否则抛出异常
        - 其余情况正常计算
    - 其他组合输入
        - 异常组合输入, 抛出异常

# 七、可行性分析和排期规划

方案主要依赖现有`paddle.diff`和`Tensor.sum`组合实现，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无

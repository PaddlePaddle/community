# paddle.pdist设计文档

| API 名称     | paddle.bitwise_right_shift<br />paddle.bitwise_left_shift |
| ------------ | --------------------------------------------------------- |
| 提交作者     | coco                                                      |
| 提交时间     | 2023-09-27                                                |
| 版本号       | V1.0                                                      |
| 依赖飞桨版本 | develop                                                   |
| 文件名       | 20230927_api_defign_for_bitwise_shift                     |

# 一、概述

## 1、相关背景

为paddle新增该API，给 Tensor 做 element wise 的算数(或逻辑)左移/右移。

## 2、功能目标

通过一个Tensor给定的bits计算另一个Tensor的的算术（或逻辑）右移/左移。

## 3、意义

飞桨支持直接对Tensor进行元素粒度的左移右移。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch中有API`torch.bitwise_right_shift(input, other, *, out=None) → Tensor`

介绍为：

```
Computes the right arithmetic shift of input by other bits. The input tensor must be of integral type. This operator supports broadcasting to a common shape and type promotion.
```

## 实现方法

从实现方法上，PyTorch是将位运算注册到element_wise系列中实现的，[代码位置](https://github.com/pytorch/pytorch/blob/844ea6408b72cf46871de85cf7a4083629a1ddd8/torch/_inductor/codegen/cpp.py#L1146-L1148)

```python
shift_right_arithmetic = _make_elementwise_binary_prim(
    "shift_right_arithmetic",
    impl_aten=torch.bitwise_right_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_logical = _not_impl  # 可见pytorch中仅支持算数位移
```

具体元素尺度的实现，[代码位置](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/common.py#L401-L405)：

```python
@staticmethod
def bitwise_right_shift(a, b):
    return f"decltype({a})({a} >> {b})"
```



## Numpy

- Parameters:

  - **x1**：array_like, int

    Input values.

  - **x2**：array_like, int

    Number of bits to remove at the right of *x1*. If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).

  - **out**：ndarray, None, or tuple of ndarray and None, optional

    A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.

  - **where**：array_like, optional

    This condition is broadcast over the input. At locations where the condition is True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array will retain its original value. Note that if an uninitialized *out* array is created via the default `out=None`, locations within it where the condition is False will remain uninitialized.

  - **kwargs：

    For other keyword-only arguments, see the [ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs).

Returns:

- **out**：ndarray, int

  Return *x1* with bits shifted *x2* times to the right. This is a scalar if both *x1* and *x2* are scalars.



相关[实现位置](https://github.com/numpy/numpy/blob/9d4c1484b96ed2b7dff49c479e9d0822a4b91f80/numpy/core/src/umath/loops_autovec.dispatch.c.src#L81-L105)

```cpp
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(@TYPE@_left_shift)
(char **args, npy_intp const *dimensions, npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
    BINARY_LOOP_FAST(@type@, @type@, *out = npy_lshift@c@(in1, in2));
#ifdef @TYPE@_left_shift_needs_clear_floatstatus
    // For some reason, our macOS CI sets an "invalid" flag here, but only
    // for some types.
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(@TYPE@_right_shift)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#ifndef NPY_DO_NOT_OPTIMIZE_@TYPE@_right_shift
    BINARY_LOOP_FAST(@type@, @type@, *out = npy_rshift@c@(in1, in2));
#else
    BINARY_LOOP {
        @type@ in1 = *(@type@ *)ip1;
        @type@ in2 = *(@type@ *)ip2;
        *(@type@ *)op1 = npy_rshift@c@(in1, in2);
    }
#endif
}
```

`npy_rshift`相关调用

```cpp
NPY_INPLACE npy_@u@@type@
npy_rshift@u@@c@(npy_@u@@type@ a, npy_@u@@type@ b)
{
    if (NPY_LIKELY((size_t)b < sizeof(a) * CHAR_BIT)) {
        return a >> b;
    }
#if @is_signed@
    else if (a < 0) {
        return (npy_@u@@type@)-1;  /* preserve the sign bit */
    }
#endif
    else {
        return 0;
    }
}
```



## Jax

算数位移

- jax.lax.**shift_right_arithmetic**(*x*, *y*)[[source\]](https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/lax.html#shift_right_arithmetic)

  Elementwise arithmetic right shift: x ≫ y.

  Parameters

  - **x** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array), [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [`bool_`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_), [`number`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.number.html#jax.numpy.number), [`bool`](https://docs.python.org/3/library/functions.html#bool), [`int`](https://docs.python.org/3/library/functions.html#int), [`float`](https://docs.python.org/3/library/functions.html#float), [`complex`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.complex.html#jax.lax.complex)]) 
  - **y** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array), [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [`bool_`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_), [`number`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.number.html#jax.numpy.number), [`bool`](https://docs.python.org/3/library/functions.html#bool), [`int`](https://docs.python.org/3/library/functions.html#int), [`float`](https://docs.python.org/3/library/functions.html#float), [`complex`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.complex.html#jax.lax.complex)]) 

  Return type[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array)

具体[实现代码](https://github.com/google/jax/blob/2d068a1caa97ebde662604bb95298cff2abb0afa/jax/experimental/jax2tf/jax2tf.py#L1748-L1760)

```python
# Note: Bitwise operations only yield identical results on unsigned integers!
# pylint: disable=protected-access
def _shift_right_arithmetic_raw(x, y):
  if x.dtype.is_unsigned:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[orig_dtype]
    x = tf.cast(x, signed_dtype)
    y = tf.cast(y, signed_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)
  else:
    return tf.bitwise.right_shift(x, y)
```

>  在算术位移过程中，如果是有符号数，则直接进行算数位移；如果是无符号数，则先转换成对应的有符号数，再进行位移（因为算术右移时常常需要补符号位，只有有符号数才能保证结果正确性），最后转回无符号数。



逻辑位移

+ jax.lax.**shift_right_logical**(*x*, *y*) [source](https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/lax.html#shift_right_logical)

  Elementwise logical right shift: x ≫ y.Parameters

  - **x** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array), [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [`bool_`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_), [`number`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.number.html#jax.numpy.number), [`bool`](https://docs.python.org/3/library/functions.html#bool), [`int`](https://docs.python.org/3/library/functions.html#int), [`float`](https://docs.python.org/3/library/functions.html#float), [`complex`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.complex.html#jax.lax.complex)]) 
  - **y** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array), [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray), [`bool_`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_), [`number`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.number.html#jax.numpy.number), [`bool`](https://docs.python.org/3/library/functions.html#bool), [`int`](https://docs.python.org/3/library/functions.html#int), [`float`](https://docs.python.org/3/library/functions.html#float), [`complex`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.complex.html#jax.lax.complex)]) 

  Return type[`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html#jax.Array)



具体[实现代码](https://github.com/google/jax/blob/2d068a1caa97ebde662604bb95298cff2abb0afa/jax/experimental/jax2tf/jax2tf.py#L1776-L1786)

```python
def _shift_right_logical_raw(x, y):
  if x.dtype.is_unsigned:
    return tf.bitwise.right_shift(x, y)
  else:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[orig_dtype]
    x = tf.cast(x, unsigned_dtype)
    y = tf.cast(y, unsigned_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)
```

> 在逻辑位移过程中，如果是无符号数，则直接进行算数位移；如果是有符号数，则先转换成无符号数，再进行位移（因为逻辑右移时常常需要补0，只有无符号数才能保证结果正确性），最后转回有符号数。





# 四、对比分析

PyTorch是将算子注册到element wise系列中，Numpy也类似地`BINARY_LOOP`来做element wise的shift操作。

同时，PyTorch与Numpy中都仅支持算术位移，不支持逻辑位移，而JAX中实现了算术位移和逻辑位移。



# 五、设计思路与实现方案

## 命名与参数设计

API的设计为`paddle.bitwise_right_shift(x, y, is_arithmetic=True)`，其余几个shift操作同理，其中 `x` 与 `y` 需要有相同的shape或者能够进行广播，且类型都必须为int；`is_arithmetic` 为bool类型，默认为 `True` 表示算术位移，当其为 `False` 时则为逻辑位移。

## API实现方案

参考`PyTorch`、`Numpy`、`JAX`中的设计，组合已有API实现功能

# 六、测试和验收的考量

测试考虑的case如下：

1. 对 `x`、`y`的 shape 和 dtype 有限制，并给出合理提示

2. 结果一致性，和 PyTorch、Numpy 结果的数值的一致性

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html?highlight=bitwise_right_shift#torch.bitwise_right_shift)

[Numpy文档](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html#numpy.right_shift)
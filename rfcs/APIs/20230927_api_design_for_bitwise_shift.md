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

具体元素尺度的实现，

[左移 cpu kernel](https://github.com/pytorch/pytorch/blob/3747aca49a39479c2c5e223b91369db5bd339cdf/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L423-L437)：

```cpp
void lshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
          if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
              (b >= max_shift)) {
            return 0;
          }
          return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a << b; });
  });
}
```

[左移 cuda kernel](https://github.com/pytorch/pytorch/blob/6e1ba79b7fdf3d66db8fb69462fb502e5006e5e7/aten/src/ATen/native/cuda/BinaryShiftOpsKernels.cu#L14-L25)

```cpp
void lshift_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cuda", [&]() {
    gpu_kernel_with_scalars(iter,
      []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
        if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) || (b >= max_shift)) {
          return 0;
        }
        return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
    });
  });
}
```

+ 可以发现，在算术左移时，kernel需要针对[两类情况进行处理](https://wiki.sei.cmu.edu/confluence/display/c/INT34-C.+Do+not+shift+an+expression+by+a+negative+number+of+bits+or+by+greater+than+or+equal+to+the+number+of+bits+that+exist+in+the+operand)：

  + `b`移动的距离大于等于当前类型的位数时（例如对int16左移16位），则直接返回0（若进行移动，编译器会在此时发生取模优化，例如左移1000位时，实际上会移动1000%16=8位，但实际上需要返回0，表示溢出）
  + `b`为负数时，在C语言标准中为"未定义行为"，认为等效于左移了无穷位，直接返回0；

  另外，kernel中用`std::make_signed_t<scalar_t>>(b)`把`b`强转为有符号数，若`b`原本就是有符号数，无影响；若`b`原本是无符号数，且最高位为0，无影响；若`b`原本是无符号数，而且较大，最高位为`1`，强转后为负数，小于0。(不过感觉即使不强转，最高位为1的无符号数应该也会令`(b >= max_shift)`为true)

  

[右移 cpu kernel](https://github.com/pytorch/pytorch/blob/3747aca49a39479c2c5e223b91369db5bd339cdf/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L494-L511)

```cpp
void rshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          // right shift value to retain sign bit for signed and no bits for
          // unsigned
          constexpr scalar_t max_shift =
              sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
          if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
              (b >= max_shift)) {
            return a >> max_shift;
          }
          return a >> b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a >> b; });
  });
}
```

[右移 cuda kernel](https://github.com/pytorch/pytorch/blob/6e1ba79b7fdf3d66db8fb69462fb502e5006e5e7/aten/src/ATen/native/cuda/BinaryShiftOpsKernels.cu#L27-L39C2)

```cpp
void rshift_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cuda", [&]() {
    gpu_kernel_with_scalars(iter,
      []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        // right shift value to retain sign bit for signed and no bits for unsigned
        constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
        if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) || (b >= max_shift)) {
          return a >> max_shift;
        }
        return a >> b;
    });
  });
}
```

+ 算术右移时，`max_shift`需要考虑最大的移动距离，有符号数最高位为符号位，故表示数值的位数实际上会少一位。
  + 有符号数时，例如`int8 x=-100`，补码为`1001,1100`，最高位为符号位，仅需要右移7位，所有的`int8`就都会变成`1111,1111`，即`-1`；
  + 无符号数时候，例如`uint8 x=200`，存储为`1100,1000`，八位均表示数值大小，需要右移8位才可以将所有的`uint8`变为`0000,0000`，即`0`；
+ 当`b`位负数这一未定义行为时，同样等效于右移无穷位，与移动`max_shift`等效，有符号数变为`-1`，无符号数变为`0`





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

`npy_lshift`[相关调用](https://github.com/numpy/numpy/blob/0032ede015c9b06f88cc7f9b07138ce35f4357ae/numpy/_core/src/npymath/npy_math_internal.h.src#L653-L662)：

```cpp
NPY_INPLACE npy_@u@@type@
npy_lshift@u@@c@(npy_@u@@type@ a, npy_@u@@type@ b)
{
    if (NPY_LIKELY((size_t)b < sizeof(a) * CHAR_BIT)) {
        return a << b;
    }
    else {
        return 0;
    }
}
```

+ 在左移时，为了防止编译器对位移的自动取模优化（例如int16类型左移100位，实际上被自动优化成左移`100%16=4`位），导致结果不为0（溢出）；

  而且这里将`b`转为`size_t`，而`size_t`是unsigned类型，所以当`b`为有符号负数时，由于补码最高位的符号位为1，所以会被转换成一个很大的正数，必然超过`sizeof(a) * CHAR_BIT`的大小，所以直接走else返回0，这里应该与`b < 0`实现了同样的效果。



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

+ 在右移时，右移的最大位数限制需要区分有符号数和无符号数：

  **此处实现与pytorch中的实现略有不同，不过结果还是等效的：pytorch中认为，有符号数最大右移位数为`n_bit-1`，而无符号数最大右移位数为`n_bit`，例如(int16最多右移15位，uint16最多右移16位，否则触发溢出，全置为符号位)；numpy中没有刻意限定符号数和无符号数的最大位移位数（例如int16和uint16的最大位移位数都是16位，都是16位才出发溢出），由于对于有符号数例如int16来说，“(pytorch)右移15位触发溢出，全部置为符号位”与“(numpy)右移15位”，两者结果是一样的，只是前者直接走溢出的else，后者真正去做了位运算而已，所以还是等效**

  

  这里的`NPY_LIKELY((size_t)b`与左移一样，隐含了`b`需要大于0。若`b`小于0，则转unsigned之后大小必然大于`sizeof(a) * CHAR_BIT`溢出，而后又根据`a`的符号位作为返回（负数溢出补码为`1111,1111,...1111`，也就是-1，正数和无符号数溢出为0）。



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


由于python层相关API的类型支持需求不合理(例如jax中的设计，unsigned转signed会溢出)，考虑下沉到cpp层实现。以右移为例，python层接口为`paddle.bitwise_right_shift`，通过参数`is_arithmetic`的设置来调用算术位移或逻辑位移的kernel，若为算术位移，则调用`_C_ops.bitwise_left_shift_arithmetic_(x, y)`，若为逻辑位移，则调用`_C_ops.bitwise_left_shift_logic_(x, y)`

cpp的kernel实现主要通过elementwise的方法，与`bitwise_and`等bitwise op设计类似，复用elementwise相关代码以支持broadcast、具体Functor的调用等。



具体行为定义：（`n_bits`表示数据类型存储位数，例如int8的`n_bits`为8，uint16的`n_bits`为16；当`y`小于0时为“未定义行为”，等效于位移超过最大位数溢出）

+ 算术位移

  + 算术左移：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回0；否则正常位移；
  + 算术右移：
    + 有符号数时：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回符号位(`a>>(n_bits-1)&1`)；否则正常位移；
    + 无符号数时：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回0；否则正常位移；

+ 逻辑位移

  + 逻辑左移：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回0；否则正常位移；

  + 逻辑右移：

    + 有符号数时：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回0；否则特殊位移：

      ```cpp
      template <typename T>
      HOSTDEVICE T logic_shift_func(const T a, const T b) {
        if (b < static_cast<T>(0) || b >= static_cast<T>(sizeof(T) * 8))
          return static_cast<T>(0);
        T t = static_cast<T>(sizeof(T) * 8 - 1);
        T mask = (((a >> t) << t) >> b) << 1;
        return (a >> b) ^ mask;
      }
      ```

      在`T mask = (((a >> t) << t) >> b) << 1;`中，先`(a >> t)`取符号位，然后`<< t`回到原位，再右移`b`后左移一位，最后与`a>>b`的结果做亦或，下面举两个例子：

      ```
      example1:
      a = 1001,1010  b = 3, 有t=7
      ((a>>t)<<t) = 1000,0000
      mask=(((a>>t)<<t)>>b)<<1 = 1110,0000
      a>>b = 1111,0011
      所以 (a>>b) ^ mask = 0001,0011
      
      example2:
      a = 0001,1010 b = 3, 有t=7
      ((a>>t)<<t) = 0000,0000
      mask=(((a>>t)<<t)>>b)<<1 = 0000,0000
      a>>b = 0000,0011
      所以 (a>>b) ^ mask = 0000，0011
      ```

    + 无符号数时：当`y`小于0，或者`y`大于等于`n_bits`时候溢出，返回0；否则正常位移；

  以上行为中，算术位移与numpy、pytorch的实现对齐；由于numpy和pytorch不支持逻辑位移，所以逻辑位移参考jax的实现思路，用numpy来进行间接实现和验证。


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

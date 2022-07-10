# paddle.frexp 设计文档

| API 名称     | paddle.frexp   |
| ------------ | ---------------------------------------- |
| 提交作者     | Ainavo                                   |
| 提交时间     | 2022-07-10                              |
| 版本号       | V1.0.0                                   |
| 依赖飞桨版本 | develop                                  |
| 文件名       | 20220706_design_for_pairwise_distance.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算相关 API，Paddle 需要扩充 API `paddle.frexp`。

## 2、功能目标

实现给出一个任意维度，任意大小的张量 `x`，会对张量当中的浮点数进行分解，得到与其对应索引相对应的尾数张量和指数张量。（两者维度和大小相同）

## 3、意义

Paddle 将可以支持使用 `paddle.frexp` 进行浮点数分解的 API，丰富其中数学相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `paddle.frexp` API，翻阅整个代码仓库也没有找到合适的解决方案，所以需要编写对应的 API。这个 API 为数学相关 API，因此需要在 `math.py` 文件添加。
- 该 API 目前在 Paddle 没有相关的参考，所以需要参考其他框架进行编写，其他框架有对应类似的解决方案，`描述业内方案调研`部分会提到。

# 三、业内方案调研

## PyTorch

在 PyTorch 中有对应的 API，`torch.frexp(input, *, out=None)`，在 PyTorch 中，该 API 的介绍为：

>Decomposes `input` into mantissa and exponent tensors such that $\text{input} = \text{mantissa} \times 2^{\text{exponent}}$.
>
>The range of mantissa is the open interval $(-1, 1)$.
>
>Supports float inputs.

### 实现方法

在实现方法上，PyTorch 是通过 C++ API 实现的，[代码位置](https://github.com/pytorch/pytorch/blob/caee732aa1632e90074a00f89b99ed5f5dbc0dbd/aten/src/ATen/native/UnaryOps.cpp#L786-L792)

相关的部分主要有：
`frexp` 函数，[位置](https://github.com/pytorch/pytorch/blob/caee732aa1632e90074a00f89b99ed5f5dbc0dbd/aten/src/ATen/native/UnaryOps.cpp#L786-L792)
```cpp
std::tuple<Tensor, Tensor> frexp(const Tensor& self) {
  Tensor mantissa = at::empty_like(self);
  Tensor exponent = at::empty_like(self, self.options().dtype(at::kInt));

  at::frexp_out(mantissa, exponent, self);
  return std::tuple<Tensor, Tensor>(mantissa, exponent);
}
```

`frexp_out` 函数，[位置](https://github.com/pytorch/pytorch/blob/caee732aa1632e90074a00f89b99ed5f5dbc0dbd/aten/src/ATen/native/UnaryOps.cpp#L794-L818)
```cpp
std::tuple<Tensor&, Tensor&> frexp_out(const Tensor& self,
                                       Tensor& mantissa, Tensor& exponent) {
  // torch.frexp is implemented for floating-point dtypes for now,
  // should add support for integral dtypes in the future.
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "torch.frexp() only supports floating-point dtypes");

  TORCH_CHECK(mantissa.dtype() == self.dtype(),
              "torch.frexp() expects mantissa to have dtype ", self.dtype(),
              " but got ", mantissa.dtype());
  TORCH_CHECK(exponent.dtype() == at::kInt,
              "torch.frexp() expects exponent to have int dtype "
              "but got ", exponent.dtype());

  auto iter = TensorIteratorConfig()
    .add_output(mantissa)
    .add_output(exponent)
    .add_input(self)
    .check_all_same_dtype(false)
    .set_check_mem_overlap(true)
    .build();
  frexp_stub(iter.device_type(), iter);

  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}
```

`frexp_stub` 函数
```cpp

```

参数表为：

**Parameters**

>  **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input tensor

**Keyword Arguments**

>  **out** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – the output tensors


## Numpy

在 Numpy 中也有对应的 API `numpy.frexp(x, [out1, out2, ]/, [out=(None, None), ]*, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'frexp'>`，对该 API 的介绍为：

>Decompose the elements of x into mantissa and twos exponent.
>
>Returns (*mantissa*, *exponent*), where `x = mantissa * 2 ** exponent`. The mantissa lies in the open interval(-1, 1), while the twos exponent is a signed integer.

但是没有找到相关的源代码。

参数表为：

- **x**：array_like

  Array of numbers to be decomposed.

- **out1**：ndarray, optional

  Output array for the mantissa. Must have the same shape as *x*.

- **out2**：ndarray, optional

  Output array for the exponent. Must have the same shape as *x*.

- **out**：ndarray, None, or tuple of ndarray and None, optional

  A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.

- **where**：array_like, optional

  This condition is broadcast over the input. At locations where the condition is True, the *out* array will be set to the ufunc result. Elsewhere, the *out* array will retain its original value. Note that if an uninitialized *out* array is created via the default `out=None`, locations within it where the condition is False will remain uninitialized.

- ***\*kwargs**

  For other keyword-only arguments, see the [ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs).

返回值：

- **mantissa**：ndarray

  Floating values between -1 and 1. This is a scalar if *x* is a scalar.

- **exponent**：ndarray

  Integer exponents of 2. This is a scalar if *x* is a scalar.


# 四、对比分析

## 共同点

- 对比 Numpy 和 PyTorch 来看，都是基于 C++ API 实现的，而且通过 Python API 调用 C++ API 实现。
- 对于输入维度不同的张量，最后输出也是对应的维度的张量。
- API 都能实现相同的功能
- 输出方式上，可以使用 `tensor.frexp() / array.frexp() ` 的方式，也可以使用 `torch.frexp / np.frexp()`

## 不同点

- PyTorch 输入的参数比较简单，而 numpy 当中有比较多的参数可以调整。
- numpy 可以实现给定条件来广播，支持多种输出模式，更加灵活。


# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.frexp(
  x: Tensor,
  out: tuple=None, optional
  name: str=None, optional
)
```


## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

实现代码：

```python
def frexp(x, out=None, name=None):
    """
    Decomposes input into mantissa and exponent tensors such that input = mantissa × 2 ^ exponent.

    Args:
        x (Tensor): The input N-D tensor, which data type should be int32, int64, float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tuple: The output Tuple of decompose to mantissa and exponent.
            Index 0: Mantissa Tensor
            Index 1: Exponent Tensor

    Examples:
        .. code-block:: Python

            import paddle
            import numpy as np

            x = paddle.arange(9.)
            print(input.numpy())
            # [0, 1, 2, 3, 4, 5, 6, 7, 8]

            mantissa, exponent = paddle.frexp(x)
            print(mantissa)
            # [0.   , 0.5  , 0.5  , 0.75 , 0.5  , 0.625, 0.75 , 0.875, 0.5  ]
            print(exponent)
            # [ 0, 1, 2, 2, 3, 3, 3, 3, 4 ]
    """

    pass
```

# 六、测试和验收的考量

测试需要考虑的 `case` 如下：

- 输出方式的正确性，主要有两种输出方式：
  1. 结果 tuple 直接作为函数返回值使用
  2. 传入一个 out 变量，将结果存储到 out 变量当中
- 输出张量的维度和大小一致性
- 输出张量结果的正确性，使用 `numpy.frexp` 作为参考
- 在动态图、静态图下都能得到正确的结果

# 七、可行性分析及规划排期

参考 numpy 和 PyTorch 的实现原理以及 参考 IEEE754 浮点数的表示方式编写对应的实现代码。

具体规划：
- 阶段一：编写对浮点数的分解的逻辑
- 阶段二：应用到对多维张量的分解
- 阶段三：书写中英文文档

# 八、影响面

增加了一个 `paddle.frexp` 的 API，使得能够实现对张量当中的浮点数进行分解。

# 名词解释

无

# 附件及参考资料

- [numpy.frexp](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html#numpy.frexp=)

- [torch.frexp](https://pytorch.org/docs/stable/generated/torch.frexp.html)

- [extracting floating-point ...](https://stackoverflow.com/questions/46093123/extracting-floating-point-significand-and-exponent-in-numpy)
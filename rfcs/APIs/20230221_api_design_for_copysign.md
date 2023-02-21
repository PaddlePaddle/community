# paddle.copysign 设计文档

| API 名称     | paddle.copysign                  |
| ------------ | -------------------------------- |
| 提交作者     | Cattidea                         |
| 提交时间     | 2023-02-21                       |
| 版本号       | V0.0.1                           |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20230221_api_design_for_copysign |

# 一、概述

## 1、相关背景

详细描述：根据两个输入逐元素地计算结果张量，其结果由第一个输入的绝对值大小及第二个输入的符号组成。此任务的目标是在 Paddle 框架中，新增 copysign API，调用路径为：paddle.copysign 和 Tensor.copysign。

## 3、意义

丰富 paddle API，增加 copysign API。

# 二、飞桨现状

目前 paddle 中存在 paddle.sign API，可以根据张量的正负返回 -1，0，-1 的值，但是没有 copysign API 的实现。

# 三、业内方案调研

## Pytorch

Pytorch 中有 API `torch.copysign(input, other, *, out=None)` ，支持广播运算，以及张量和浮点数输入：

```
Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise
```

$$
out_i=\begin{cases}
-|input_i| & if other_i<=-0.0 \\
|input_i| & if other_i >=0.0
\end{cases}
$$

```
Parameters:
- iput(Tensor)-magnitudes.
- other (Tensor or Number) – contains value(s) whose signbit(s) are applied to the magnitudes in input.
Keyword Arguments:
- out (Tensor, optional) – the output tensor.
```

官方文档链接为：https://pytorch.org/docs/stable/generated/torch.copysign.html

## Tensorflow

在 Tensorflow 中没有 copysign 的实现，但是有 [signbit](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/signbit) 这个 API 与之相关，可以考虑使用 tf.experimental.numpy.signbit API 与其他 API 组合的形式，实现 copysign API 的功能。

## Numpy

Numpy 中有 API `numpy.copysign(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'copysign'>` 按元素将 $x_1$ 符号更改成 $x_2$ 的符号，如果 $x_2$ 是标量，将其符号复制到 $x_1$ 的所有元素。

Parameters:

- x1(array_like):Values to change the sign of.
- x2(array_like):The sign of x2 is copied to x1.If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).
- out(ndarray, None, or tuple of ndarray and None, optional): A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
- where(array_like, optional):This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. Elsewhere, the out array will retain its original value. Note that if an uninitialized out array is created via the default out=None, locations within it where the condition is False will remain uninitialized.
- \*\*kwargs:For other keyword-only arguments.
  Returns:

- out(ndarray or scalar):The values of x1 with the sign of x2. This is a scalar if both x1 and x2 are scalars.

官方文档链接为：https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy-copysign

### 实现方法

代码如下：

Pytorch 中具体使用 `python` 实现，[代码](https://cs.github.com/pytorch/pytorch/blob/4d753b50451607b3314f827993df7e5527f0c0a7/torch/_refs/__init__.py#L1033)如下：

```python
def copysign(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    if isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        msg = "Expected divisor (b) to be on the same device ({0}) as dividend (a), but it is found on {1}!".format(
            a.device, b.device
        )
        raise RuntimeError(msg)
    return where(signbit(b), neg(abs(a)), abs(a))
```

Numpy 中的 copysign API 是通过 C++ 代码实现的，详细代码如下所示：

```C++
identity = NULL;
if (0 && identity == NULL) {
    return -1;
}
f = PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
    copysign_functions, copysign_data, copysign_signatures, 4,
    2, 1, PyUFunc_None, "copysign",
    DOC_NUMPY_CORE_UMATH_COPYSIGN, 0, NULL, identity
);
if (0) {
    Py_DECREF(identity);
}
if (f == NULL) {
    return -1;
}

PyDict_SetItemString(dictionary, "copysign", f);
Py_DECREF(f);
```

# 四、对比分析

Pytorch 通过 python 现有的 API 组合实现，代码通俗易懂，整体设计较为清晰。  
Numpy 中通过调用底层的 C++代码实现，具体逻辑不详细展开。  
TensorFlow 中没有 copysign API 的实现方式。  
因此，paddle 中 copysign 主要参考 Pytorch 的实现方式，利用已有的 API 组合实现。

# 五、方案设计

## 命名与参数设计

API 设计为`paddle.copysign(x, y, name=None)`和`paddle.Tensor.copysign(x, y, name=None)`

- x(Tensor):输入。
- y(Tensor or np.ndarray):包含张量或数组或标量。

## 底层 OP 设计

使用已有 API 进行组合，不再单独设计底层 OP。

## API 实现方案

- 首先判断输入 x 和 y 的类型，判断输入 input 是否为张量，输入参数 y 是否为数组或者张量。
- 由于 Pytorch 中的代码判断正负用到 torch.signbit API，**主要是针对-0 和+0 的判断**，-0 对应负号，+0 对应正号，但是 paddle 目前不支持-0 和+0 的判断，因此目前不考虑-0 的情况，在 paddle 中可用`[x>=0]`作为判断条件。
- 通过 `[x>=0]` 作为判断条件，获得 True 和 False，然后通过 paddle.where 根据条件，分别赋予 paddle.abs(x) 不同的符号。

# 六、测试和验收的考量

测试考虑的 case 如下：

- 编程范式场景：覆盖静态图和动态图测试场景。
- 硬件场景：覆盖 CPU 和 GPU 测试场景。
- x：Tensor，支持 float16， float32，int。
- y：Tensor or np.ndarray 支持 float16，float32， int。
- y 取+0 和 -0 时 paddle.copysign 的正确性（注：-0 现在没找到好的解决方案，因此当 y 取 $\pm0$ 时，都按大于等于 0 处理）。
- 计算精度：前向计算，和 numpy 实现的函数对比结果；反向计算，由 Python 组合的新增 API 无需验证反向计算。
- 验证广播机制的正常。

# 七、可行性分析及规划排期

方案主要依赖 paddle 现有 API 组合而成，并自行实现核心算法。

# 八、影响面

为独立新增 API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

Pytorch [相关文档](https://pytorch.org/docs/stable/generated/torch.copysign.html)

Numpy [相关文档](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy-copysign)

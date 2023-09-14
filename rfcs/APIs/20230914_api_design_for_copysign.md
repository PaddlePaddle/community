# paddle.copysign 设计文档

| API 名称     | paddle.copysign                  |
| ------------ | -------------------------------- |
| 提交作者     | enkilee                         |
| 提交时间     | 2023-09-14                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20230914_api_design_for_copysign |

# 一、概述

## 1、相关背景

详细描述：根据两个输入逐元素地计算结果张量，其结果由第一个输入的绝对值大小及第二个输入的符号组成。此任务的目标是在 Paddle 框架中，新增 copysign API，调用路径为：paddle.copysign、paddle.copysign_、Tensor.copysign、Tensor.copysign_。

## 3、意义

丰富 paddle API，增加 copysign API。

# 二、飞桨现状

目前 paddle 中存在 paddle.sign API，可以根据张量的正负返回 -1，0，-1 的值，但是没有 copysign API 的实现。

# 三、业内方案调研

## 3.1 Pytorch

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
- input(Tensor)-magnitudes.
- other (Tensor or Number) – contains value(s) whose signbit(s) are applied to the magnitudes in input.
Keyword Arguments:
- out (Tensor, optional) – the output tensor.

```

官方文档链接为：https://pytorch.org/docs/stable/generated/torch.copysign.html?highlight=copysign#torch.copysign

### 3.1.1 实现代码：

- 前向逻辑代码通过 std::copysign 实现的，[代码位置](https://github.com/pytorch/pytorch/blob/main/c10/util/copysign.h)

```cpp
  template <typename T, typename U>
  inline auto copysign(const T& a, const U& b) {
  return std::copysign(a, b);
  }

  // Implement copysign for half precision floats using bit ops
  // Sign is the most significant bit for both half and bfloat16 types
  inline c10::Half copysign(c10::Half a, c10::Half b) {
  return c10::Half((a.x & 0x7fff) | (b.x & 0x8000), c10::Half::from_bits());
  }

  inline c10::BFloat16 copysign(c10::BFloat16 a, c10::BFloat16 b) {
  return c10::BFloat16(
      (a.x & 0x7fff) | (b.x & 0x8000), c10::BFloat16::from_bits());
  }
  }
```

- 反向逻辑代码位于 torch/csrc/autograd/FunctionsManual.cpp 中的函数[copysign_tensor_self_backward](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp#L94)

  ```C++
    Tensor copysign_tensor_self_backward(
        const Tensor& grad,
        const Tensor& self,
        const Tensor& result) {
      auto ratio = result / self;
      ratio.masked_fill_(self == 0, 0);
      return grad * ratio;
    }
  ```

## 3.2 Tensorflow

在 Tensorflow 中没有 copysign 的实现。

## 3.3 Scipy

在 Scipy 中没有 copysign 的实现。

## 3.4 Numpy

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

### 3.4.1 实现代码：
**Numpy** 中的 copysign API 是通过 C++ 代码实现的，详细代码如下所示：

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

- PyTorch 是通过 std::copysign 实现的实现，使用 Python 调用 C++ API 对应的接口。 Paddle 也可以基于 C++ API 实现。

# 五、设计思路与实现方案

## 命名与参数设计

添加 API

```python
paddle.copysign(
    x: Tensor,
    y: Tensor or Number,
    name: str=None
)
paddle.copysign_(
    x: Tensor,
    y: Tensor or Number,
    name: str=None
)

paddle.Tensor.copysign(
    x: Tensor,
    y: Tensor or Number,
    name: str=None
)

paddle.Tensor.copysign_(
    x: Tensor,
    y: Tensor or Number,
    name: str=None
)
```

## 底层OP设计

底层增加 copysign OP。

前向逻辑:

- 参考 pytorch 基于 C++ <cmath> 库中的 `copysign` 函数，直接实现。

反向逻辑:

- 当输入 `x` 为 `0`时，反向梯度都为 `0`。
- 当输入 `x` 不为 `0`时，通过计算结果 `out` 除以 `x` 得到的比值，并乘 `x` 的梯度。
- 当输入 `y` 为 `Tensor`时，`y` 的梯度为 `0`。

## API实现方案

通过调研发现，需要
1. `paddle/phi/api/yaml/op_compat.yaml`、`paddle/phi/api/yaml/ops.yaml` 添加算子 Copysign。
2. `paddle/phi/infermeta/binary.cc`、`paddle/phi/infermeta/binary.h` 添加算子 CopysignInferMeta。
3. `paddle/phi/kernels/cpu/` 目录下添加 `copysign_kernel.cc`文件。
4. `paddle/phi/kernels/gpu/` 目录下添加 `copysign_kernel.cu`文件。
5. `paddle/phi/kernels/impl/` 目录下添加 `copysign_kernel_impl.h`文件, C++实现代码。
6. `paddle/phi/kernels/`目录下添加 `copysign_kernel.h`文件。
7. `python/paddle/__init__.py` 添加 copysign API，以支持 Tensor.copysign 的调用方式。
8. `python/paddle/tensor/math.py` 添加Python 实现代码 & 英文 API 文档。
9. `python/paddle/fluid/tests/unittests/` 目录下添加单测文件 `test_copysign_op.py`。


# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 编程范式场景：覆盖静态图和动态图测试场景。
- 硬件场景：覆盖 CPU 和 GPU 测试场景。
- 数据类型检验：
  - x 要求为 paddle.Tensor，支持 uint8, float16、float32、float64、int32、int64、bool。
  - y 要求为 paddle.Tensor，Number 支持 uint8, float16、float32、float64、int32、int64、bool。
- y 取 +0 和 -0 时 paddle.copysign 的正确性。
- 结果的正确性：
  - 前向计算：`paddle.copysign` 的计算结果和 `np.copysign` 一致。
  - 反向计算：`paddle.copysign` 的计算反向传播所得到的梯度与使用 numpy 手动计算的结果一致。

# 七、可行性分析和排期规划

方案主要依赖现有 Paddle 而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

Pytorch [相关文档](https://pytorch.org/docs/stable/generated/torch.copysign.html)

Numpy [相关文档](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy-copysign)

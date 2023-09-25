# paddle.copysign 设计文档

| API 名称     | paddle.signbit                  |
| ------------ | -------------------------------- |
| 提交作者     | PommesPeter                             |
| 提交时间     | 2023-09-25                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20230925_api_defign_for_signbit.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，Paddle 需要扩充 API `paddle.signbit`。

## 2、功能目标

测试输入的每个元素是否设置了其符号位（小于零）。

- `paddle.signbit` 作为独立的函数调用
- `Tensor.signbit(x)` 做为 Tensor 的方法使用

## 3、意义

为 Paddle 增加测试输入的每个元素是否设置了其符号位，丰富 `paddle` 中符号相关的 API。

# 二、飞桨现状

目前 Paddle 缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.signbit(input, *, out=None) → Tensor` 以及对应的 `torch.Tensor.signbit`

在 PyTorch 中介绍为：

> Tests if each element of input has its sign bit set or not.

参数表如下：
- Parameters:
  - input (Tensor) – the input tensor.

- Keyword Arguments:
  - out (Tensor, optional) – the output tensor.


**前向实现：**

从实现方法上，PyTorch 是通过 C++ 实现的，[CPU Kernel 代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp#L321)

```cpp
static void signbit_kernel(TensorIteratorBase& iter){
  // NOTE: signbit does not always support integral arguments.
  AT_DISPATCH_SWITCH(iter.input_dtype(), "signbit_cpu",
      AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
        cpu_kernel(iter, [](scalar_t a) -> bool { return c10::is_negative(a); });
      })
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        cpu_kernel(iter, [](scalar_t a) -> bool { return std::signbit(opmath_t{a}); });
      })
    );
}
```

在 c10 中，[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/c10/util/math_compat.h#L118-L128)：

```cpp
namespace std {

// Note: std::signbit returns true for negative zero (-0), but this
// implementation returns false.
inline bool signbit(float x) {
  return x < 0;
}
inline bool signbit(double x) {
  return x < 0;
}
inline bool signbit(long double x) {
  return x < 0;
}
...
} // namespace std
```

[cuda kernel代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/native/cuda/UnarySignKernels.cu#L76-L88)

```cpp
namespace at::native {

void signbit_kernel_cuda(TensorIteratorBase& iter){
  // NOTE: signbit does not always support integral arguments.
  if (at::isIntegralType(iter.input_dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.input_dtype(), "signbit_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return is_negative(a); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return signbit(opmath_t{a}); });
    });
  }
}

REGISTER_DISPATCH(signbit_stub, &signbit_kernel_cuda);

} // namespace at::native
```

**反向实现：**

无反向实现。

## TensorFlow

Tensorflow 是通过 Python API 的方式实现 `signbit`。

[代码位置](https://github.com/tensorflow/tensorflow/blob/v2.13.0/tensorflow/python/ops/numpy_ops/np_math_ops.py#L658-L666)

```python
@np_utils.np_doc('signbit')
def signbit(x):

  def f(x):
    if x.dtype == dtypes.bool:
      return array_ops.fill(array_ops.shape(x), False)
    return x < 0

  return _scalar(f, x)
```
Unsupported arguments: `out`, `where`, `casting`, `order`, `dtype`, `subok`, `signature`, `extobj`.

## Numpy

`numpy.signbit(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'signbit'>`

> Returns element-wise True where signbit is set (less than zero).

实现代码

```cpp
static int
_signbit_set(PyArrayObject *arr)
{
    static char bitmask = (char) 0x80;
    char *ptr;  /* points to the npy_byte to test */
    char byteorder;
    int elsize;

    elsize = PyArray_DESCR(arr)->elsize;
    byteorder = PyArray_DESCR(arr)->byteorder;
    ptr = PyArray_DATA(arr);
    if (elsize > 1 &&
        (byteorder == NPY_LITTLE ||
         (byteorder == NPY_NATIVE &&
          PyArray_ISNBO(NPY_LITTLE)))) {
        ptr += elsize - 1;
    }
    return ((*ptr & bitmask) != 0);
}

```

# 四、对比分析

PyTorch 底层调用 C++ `std::signbit` 实现，Tensorflow 通过 Python API 组合实现。两者均不能进行反向传播。

# 五、设计思路与实现方案

## 命名与参数设计

添加 Python API:

```python
paddle.signbit(
    x: Tensor,
    name: str=None
)
```

参数表：
- x: (Tensor) 输入的 tensor。数据类型支持 `float16`、`float32`、`float64`、`uint8`、`int8`、`int16`、`int32`、`int64`、`bfloat16`
- name: (str) 算子的名称。

## 底层 OP 设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案



1. 配置算子的yaml，注意配置inplace
2. 实现`CopySignInferMeta`，在调用kernel之前计算好`out`的`shape`和`dtype`
3. 实现`CopySignKernel`的CPU和GPU代码以及forward、backward
4. 封装Python的API，支持动态图和静态图，编写文档
5. 编写单测

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**：常规覆盖动态图和静态图的测试场景
- **硬件场景**：常规需覆盖 CPU、GPU 两种测试场景
- **参数组合场景**：需要对全部入参进行参数有效性和边界值测试；同时可选参数也需有相应的测试覆盖；输入输出的容错性与错误提示信息
- **计算精度**：需要保证前向计算、反向计算的精度正确性
  - 前向计算：通过 numpy 实现的函数的对比结果
- **维度测试**：Paddle API 支持的最低维度为 0 维，单测中应编写相应的 0 维尺寸测试 case
- **边界测试**：x 为 0、+0、-0 时，测试与 numpy 结果的一致性

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.signbit.html)

[Numpy 文档](https://numpy.org/doc/stable/reference/generated/numpy.signbit.html)
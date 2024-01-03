# paddle.copysign 设计文档

| API 名称     | paddle.copysign                  |
| ------------ | -------------------------------- |
| 提交作者     | coco                             |
| 提交时间     | 2023-09-14                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20230914_api_defign_for_copysign |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，Paddle需要扩充API，调用路径为：

- paddle.copysign 作为独立的函数调用，非 inplace
- paddle.copysign_，作为独立的函数，inplace 地修改输入；
- Tensor.copysign做为 Tensor 的方法使用，非 inplace;
- Tensor.copysign_做为 Tensor 的方法使用， inplace 修改输入；

## 2、功能目标

根据两个输入逐元素地计算结果张量，其结果由第一个输入的绝对值大小及第二个输入的符号组成。

## 3、意义

飞桨支持直接通过张量进行批量正负符号复制

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch中有API `torch.copysign(input, other, *, out=None) → [Tensor]` 以及对应的`torch.Tensor.copysign`

在PyTorch中介绍为：

```
Create a new floating-point tensor with the magnitude of input and the sign of other, elementwise.
 
Supports broadcasting to a common shape, and integer and float inputs.
```

## 实现方法

从实现方法上，PyTorch是通过c++实现的，[CPU kernel代码位置](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L1148-L1158)

```cpp
void copysign_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cpu", [&]() {
    cpu_kernel_vec(iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        return c10::copysign(a, b);
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
        return a.copysign(b);
      });
  });
}
```

在c10 namespace中，[代码位置](https://github.com/pytorch/pytorch/blob/main/c10/util/copysign.h#L12-L15)：

```cpp
namespace c10 {

// Note: Explicit implementation of copysign for Half and BFloat16
// is needed to workaround g++-7/8 crash on aarch64, but also makes
// copysign faster for the half-precision types
template <typename T, typename U>
inline auto copysign(const T& a, const U& b) {
  return std::copysign(a, b);
}
...
} // namespace c10
```

[cuda kernel代码位置](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CopysignKernel.cu#L23-L29)

```cpp
namespace at::native {

void copysign_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return c10::cuda::compat::copysign(a, b);
    });
  });
}

REGISTER_DISPATCH(copysign_stub, &copysign_kernel_cuda);

} // namespace at::native
```

namespace中的`copysign`调用，[代码位置](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAMathCompat.h#L46-L65)

```cpp
__MATH_FUNCTIONS_DECL__ float copysign(float x, float y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysignf(x, y);
#else
  // std::copysign gets ICE/Segfaults with gcc 7.5/8 on arm64
  // (e.g. Jetson), see PyTorch PR #51834
  // This host function needs to be here for the compiler but is never used
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}
__MATH_FUNCTIONS_DECL__ double copysign(double x, double y) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
  return ::copysign(x, y);
#else
  // see above
  TORCH_INTERNAL_ASSERT(
      false, "CUDAMathCompat copysign should not run on the CPU");
#endif
}
```

方法都是底层cpp调用copysign函数



**反向backward:**

算子配置[代码位置](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml#L474-L481C28)

```yaml
- name: copysign.Tensor(Tensor self, Tensor other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
  other: zeros_like(other)
  result: copysign_tensor_self_backward(self_t, self_p, result)

- name: copysign.Scalar(Tensor self, Scalar other) -> Tensor
  self: copysign_tensor_self_backward(grad, self, result)
  result: auto_element_wise
```

backward 反向[代码位置](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp#L94-L101)

```cpp
Tensor copysign_tensor_self_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result) {
  auto ratio = result / self;
  ratio.masked_fill_(self == 0, 0);
  return grad * ratio;
}
```

## TensorFlow

无`copysign`实现

## Numpy

numpy.**copysign**(*x1*, *x2*, */*, *out=None*, ***, *where=True*, *casting='same_kind'*, *order='K'*, *dtype=None*, *subok=True*[, *signature*, *extobj*]) *= <ufunc 'copysign'>*

Change the sign of x1 to that of x2, element-wise.If *x2* is a scalar, its sign will be copied to all elements of *x1*.

### 实现方法

先模板生成函数，底层cpp调用实现[代码位置](https://github.com/numpy/numpy/blob/main/numpy/core/src/umath/loops.c.src#L1213-L1221)

```
NPY_NO_EXPORT void
@TYPE@_copysign(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    BINARY_LOOP {
        const @type@ in1 = *(@type@ *)ip1;
        const @type@ in2 = *(@type@ *)ip2;
        *((@type@ *)op1)= npy_copysign@c@(in1, in2);
    }
}
```

实际调用cpp的math库[代码位置](https://github.com/numpy/numpy/blob/main/numpy/core/include/numpy/npy_math.h#L199)

```cpp
#include <math.h>

...
#define npy_copysign copysign
...
```



# 四、对比分析

PyTorch和Numpy实现方式基本一致，都是底层调用cpp的math库实现`copysign`，PyTorch可进行backward。

paddle的promotion机制暂时正在建设中，故现仅考虑输入的两个元素的类型相同的情况，下面是对竞品面对不同输入类型的行为记录（pytorch和numpy均不支持complex类型）：

| x                | y                | np.copysign(x,y)/torch.copysign(x,y) | grad_x(grad_y) |
| ---------------- | ---------------- | ------------------------------------ | -------------- |
| np.uint8         | np.uint8         | np.float16                           | /              |
| torch.uint8      | torch.uint8      | torch.float32                        | /              |
| np.int8          | np.int8          | np.float16                           | /              |
| torch.int8       | torch.int8       | torch.float32                        | /              |
| np.int16         | np.int16         | np.float32                           | /              |
| torch.int16      | torch.int16      | torch.float32                        | /              |
| np.int32         | np.int32         | np.float64                           | /              |
| torch.int32      | torch.int32      | torch.float32                        | /              |
| np.int64         | np.int64         | np.float64                           | /              |
| torch.int64      | torch.int64      | torch.float32                        | /              |
| np.float16       | np.float16       | np.float16                           | /              |
| torch.float16    | torch.float16    | torch.float16                        | torch.float16  |
| np.float32       | np.float32       | np.float32                           | /              |
| torch.float32    | torch.float32    | torch.float32                        | torch.float32  |
| np.float64       | np.float64       | np.float64                           | /              |
| torch.float64    | torch.float64    | torch.float64                        | torch.float64  |
| np.complex64     | np.complex64     | /                                    | /              |
| torch.complex64  | torch.complex64  | /                                    | /              |
| np.complex128    | np.complex128    | /                                    | /              |
| torch.complex128 | torch.complex128 | /                                    | /              |
| np.bool          | np.bool          | np.float16                           | /              |
| torch.bool       | torch.bool       | torch.float32                        | /              |
| torch.bfloat16   | torch.bfloat16   | torch.bfloat16                       | torch.bfloat16 |

+ 可以发现，在整型输入时，numpy和pytorch的行为略有不同：pytorch面对整型输入，均保持`float32`作为输出，而numpy在整型输入时，仅当dtype为`int16`时，输出的dtype与pytorch对齐（均为`float32`）。
+ 另外，pytorch不支持整型(包括bool)的反向传播（但是paddle目前似乎并未对此作限制）。对于浮点数，输入和输出类型保持一致。

# 五、设计思路与实现方案

## 命名与参数设计

API的设计为:

- paddle.copysign(x, y, name=None) 作为独立的函数调用，非 inplace;
- paddle.copysign_(x, y, name=None)，作为独立的函数，inplace 地修改输入;
- Tensor.copysign(y, name=None)做为 Tensor 的方法使用，非 inplace;
- Tensor.copysign_(y, name=None)做为 Tensor 的方法使用， inplace 修改输入;

其中

+ x(Tensor) - 需要取用绝对值作为输出数值部分的 Tensor , 支持 `bool`、`float16`、`float32`、`float64`、`uint8`、`int8`、`int16`、`int32`、`int64`、`bfloat16`
+ y(Tensor | Number) - 为 Tensor 时，shape 需要与 x 相同，或者可广播成 x.shape，支持 `bool`、`float16`、`float32`、`float64`、`uint8`、`int8`、`int16`、`int32`、`int64`、`bfloat16`；为 Number 时，支持 `bool`、`int`、`float`

## 底层OP设计

参考`elementwise_compute`类型的其他op，支持broadcast。

**需要进一步确认的点：**

+ 竞品均直接采用cpp标准库中的`std::copysign`来实现Functor，然而这个库函数在接收整型输入时，自动会进行promotion为浮点数：

  ```cpp
  #ifndef __CORRECT_ISO_CPP11_MATH_H_PROTO_FP
  
  constexpr float
  copysign(float __x, float __y)
  { return __builtin_copysignf(__x, __y); }
   
  constexpr long double
  copysign(long double __x, long double __y)
  { return __builtin_copysignl(__x, __y); }
  #endif
   
  #ifndef __CORRECT_ISO_CPP11_MATH_H_PROTO_INT
  template<typename _Tp, typename _Up>
  constexpr typename __gnu_cxx::__promote_2<_Tp, _Up>::__type
  copysign(_Tp __x, _Up __y)
  {
  typedef typename __gnu_cxx::__promote_2<_Tp, _Up>::__type __type;
  return copysign(__type(__x), __type(__y));
  }
  #endif
  ```

  而且numpy和pytorch遇到整型输入，得到输出dtype常常不同，需要确定paddle的实现。

+ 反向传播的dtype是否一直保持不变？如果是，那么遇到dtype为整型的时（paddle没有严格限制反向传播过程的dtype不能为整型，pytorch有强制限制反向传播过程不能为整型），求梯度会有f(x,y)得到float类型，就变了。paddle目前支持反向传播过程中数据类型发生变化吗？

单从功能上来看，`copysign`实现的逻辑比较简单：取第一个变量的绝对值的大小，取第二个变量符号，两者拼接。感觉从功能上来看，可以不拘泥于跟竞品一样调用标准库的`std::copysign`，而是直接在Functor中判断来实现，而且输入和输出dtype保持相同。

## API实现方案

1. 配置算子的yaml，注意配置inplace
2. 实现`CopySignInferMeta`，在调用kernel之前计算好`out`的`shape`和`dtype`
3. 实现`CopySignKernel`的CPU和GPU代码以及forward、backward
4. 封装Python的API，支持动态图和静态图，编写文档
5. 编写单测

# 六、测试和验收的考量

测试考虑的case如下：

+ **编程范式场景**：常规覆盖动态图和静态图的测试场景

+ **硬件场景**：常规需覆盖 CPU、GPU 两种测试场景
+ **参数组合场景**：常规覆盖 API 的全部入参，需要对全部入参进行参数有效性和边界值测试，同时可选参数也需有相应的测试覆盖
+ **计算精度**：需要保证前向计算、反向计算的精度正确性
  + 前向计算：通过 numpy 实现的函数的对比结果
  + 反向计算：通过 numpy 推导，计算反向结果的正确性
+ **维度测试**：Paddle API 支持的最低维度为 0 维，单测中应编写相应的 0 维尺寸测试 case
+ **边界测试**：y为0、+0、-0时，测试与numpy结果的一致性

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.copysign.html?highlight=copysign#torch.copysign)

[Numpy文档](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy-copysign)
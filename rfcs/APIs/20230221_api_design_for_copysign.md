# paddle.copysign 设计文档

| API 名称     | paddle.copysign                  |
| ------------ | -------------------------------- |
| 提交作者     | Cattidea                         |
| 提交时间     | 2023-04-16                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20230416_api_design_for_copysign |

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
- input(Tensor)-magnitudes.
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

**Pytorch** 中具体使用 `C++` 实现具体步骤如下：

- 首先，引用 C++ <cmath> 库中的 `__builtin_copysignf__(x, y)` 实现 copysign 函数功能，[具体代码](https://github.com/pytorch/pytorch/blob/main/c10/util/math_compat.h)如下：

      ````C++
      #pragma once

      #include <cmath>

      // Android NDK platform < 21 with libstdc++ has spotty C++11 support.
      // Various hacks in this header allow the rest of the codebase to use
      // standard APIs.
      #if (defined(__ANDROID__) && __ANDROID_API__ < 21 && defined(__GLIBCXX__)) || \
          defined(__NEWLIB__)
      #include <stdexcept>
      // ....
      inline float copysign(float x, float y) {
      return __builtin_copysignf(x, y);
      }
      //...
      // Convoluted definition of these binary functions for overloads other than
      // (float,float) and (double,double).  Using a template from __gnu_cxx
      // is dirty, but this code is only enabled on a dead platform, so there
      // shouldn't be any risk of it breaking due to updates.
      template <typename T, typename U>
      typename __gnu_cxx::__promote_2<T, U>::__type copysign(T x, U y) {
      typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
      return copysign(type(x), type(y));
      }

- 然后在 'c10/util/' 路径下创建头文件[copysign.h](https://github.com/pytorch/pytorch/blob/main/c10/util/copysign.h) 实现对 `copysign` 函数的调用，具体如下：

  ```C++
  #pragma once

  #include <c10/util/BFloat16.h>
  #include <c10/util/Half.h>
  #include <c10/util/math_compat.h>

  namespace c10 {

  // Note: Explicit implementation of copysign for Half and BFloat16
  // is needed to workaround g++-7/8 crash on aarch64, but also makes
  // copysign faster for the half-precision types
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

  } // namespace c10
  ```

  同时在 'c10\cuda\CUDAMathCompat.h' 路径下添加 `copysign` 使得编译成功（注释是这么说的~），[代码](https://github.com/pytorch/pytorch/blob/8f1c3c68d3aba5c8898bfb3144988aab6776d549/c10/cuda/CUDAMathCompat.h#L46)如下：

  ```C++
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

- 然后，在 'pytorch/aten/src/ATen/native/cuda' 路径下实现对 'c10/ 下的 `copysign` 函数的调用，[具体](https://github.com/pytorch/pytorch/blob/8f1c3c68d3aba5c8898bfb3144988aab6776d549/aten/src/ATen/native/cuda/CopysignKernel.cu#L23)如下:

  ```C++
  #define TORCH_ASSERT_NO_OPERATORS
  #include <ATen/Dispatch.h>
  #include <ATen/native/DispatchStub.h>
  #include <ATen/native/cuda/Loops.cuh>
  #include <ATen/native/TensorIterator.h>
  #include <ATen/native/BinaryOps.h>

  #if defined(__CUDACC__)
  #include <cuda.h>
  #include <cuda_fp16.h>
  #include <c10/cuda/CUDAMathCompat.h>
  #elif defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #include <c10/hip/HIPMathCompat.h>
  #endif

  // NOTE: CUDA on Windows requires that the enclosing function
  // of a __device__ lambda not have internal linkage.

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

- copysign op 的反向逻辑代码：
  当参数 `y` 为 `Tensor` 或 `Number` 类型时，copysign 的反向逻辑代码分别位于 pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml 的 yaml 文件自动生成：

  ```yaml
  - name: copysign.Tensor(Tensor self, Tensor other) -> Tensor
    self: copysign_tensor_self_backward(grad, self, result)
    other: zeros_like(other)
    result: copysign_tensor_self_backward(self_t, self_p, result)

  - name: copysign.Scalar(Tensor self, Scalar other) -> Tensor
    self: copysign_tensor_self_backward(grad, self, result)
    result: auto_element_wise
  ```

  其中：当 `y` 为 `Tensor` 时，反向逻辑的具体实现代码位于 pytorch/torch/csrc/autograd/FunctionsManual.cpp 中的函数[copysign_tensor_self_backward](https://github.com/pytorch/pytorch/blob/72daadef2c063d160605e1fb7d84eeeccd55510f/torch/csrc/autograd/FunctionsManual.cpp#L94)

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

  当 `y` 为 `Number` 时，通过 `auto_element_wise` 函数返回结果。

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

Pytorch 通过底层 C++ 代码实现。
Numpy 中通过调用底层的 C++代码实现，具体逻辑不详细展开。
TensorFlow 中没有 copysign API 的实现方式。

# 五、方案设计

## 命名与参数设计

API 设计为`paddle.copysign(x, y, name=None)`和`paddle.Tensor.copysign(x, y, name=None)`

- x(Tensor):预计支持 uint8、int、float、bool 等 Tensor 数据类型。
- y(Tensor or Number)：预计支持 uint8、int、float、bool 等 Tensor/Number 数据类型。

## 底层 OP 设计

前向逻辑目前两种策略：

- 参考 torch 基于 C++ <cmath> 库中的 `copysign` 函数，直接实现。
- 自己实现 `copysign` 底层逻辑。

反向逻辑实现(参考 torch 中对应的实现)：

- 当输入 `x` 为 `0`时，反向梯度都为 `0`。
- 当输入 `x` 不为 `0`时，通过计算结果 `out` 除以 `x` 得到的比值，并乘 `x` 的梯度。
- 当输入 `y` 为 `Tensor`时，`y` 的梯度为 `0`。

## API 实现方案

参考 pytorch 的实现是调用 C++ <cmath> 库中的 copysign 函数：

- 首先，实现 copysign 的 InferMeta 函数，位于 paddle/phi/infermeta/binary.h 中，通过输入 `x`， 推断输出 `out` 的 `shape` 和 `dtype`；
- 然后，配置 Yaml 文件；
- 分别在 CPU 和 GPU 以及其他设备下实现 copysign kernel 的前向、反向逻辑；
- 封装成 python API，判断输入数据类型，实现动态图、静态图的分支；
- 最后补充单测。

# 六、测试和验收的考量

测试考虑的 case 如下：

- 编程范式场景：覆盖静态图和动态图测试场景。
- 硬件场景：覆盖 CPU 和 GPU 测试场景。
- 数据类型检验：
  - x 要求为 paddle.Tensor，支持 uint8, float16、float32、float64、int32、int64、bool。
  - y 要求为 paddle.Tensor，Number 支持 uint8, float16、float32、float64、int32、int64、bool。
- y 取 +0 和 -0 时 paddle.copysign 的正确性。
- 结果的正确性：
  - 前向计算：`paddle.copysign` 的计算结果和 `np.copysign` 一致。
  - 反向计算：`paddle.copysign` 的计算反向传播所得到的梯度与使用 numpy 手动计算的结果一致。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点，可在周期内完成。

# 八、影响面

为独立新增 API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

Pytorch [相关文档](https://pytorch.org/docs/stable/generated/torch.copysign.html)

Numpy [相关文档](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy-copysign)

copysign 相关 [pr](https://github.com/pytorch/pytorch/pull/46396)

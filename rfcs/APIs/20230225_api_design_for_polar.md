# paddle.polar 设计文档

| API 名称     | polar                            |
| ------------ | ------------------------------------ |
| 提交作者     | catcatjam                             |
| 提交时间     | 2023-02-25                          |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop                              |
| 文件名       | 20230225_api_design_for_polar.md |

# 一、概述

## 1、相关背景

Paddle 需要新增 API：paddle.polar 和 Tensor.polar


## 2、功能目标

通过输入模和相位角，elementwise 构造复数 tensor。

## 3、意义

为 paddle 框架提供另一种构造复数tensor的方法。

# 二、飞桨现状

飞桨内有paddle.complex 函数，用户可以通过输入实部和虚部构造复数 tensor。

# 三、业内方案调研

## Pytorch

Pytorch 中有 `torch.polar(abs, angle, *, out=None)` API

`torch.polar`要求提供的参数如下：
| Name       | Type  | Description                                                  |
| ---------- | ----- | ------------------------------------------------------------ |
| abs       | Tensor   | The absolute value the complex tensor. Must be float or double  |
| angle        | Tensor | The angle of the complex tensor. Must be same dtype as `abs`.|

> 官方文档链接为：https://pytorch.org/docs/stable/generated/torch.polar.html


## Tensorflow

tensorflow 没有提供以模(modulus)和相位角(angle)直接创建复数tensor的API，只有提供与`paddle.complex`类似的`tf.dtypes.complex(real, imag, name=None)`
`tf.dtypes.complex`要求提供的参数如下：
| Name       | Type  | Description                                                  |
| ---------- | ----- | ------------------------------------------------------------ |
| real       | Tensor   | The real part of the complex number  |
| imag        | Tensor | The imaginary part of the complex number|

官方文档链接为：https://www.tensorflow.org/api_docs/python/tf/dtypes/complex

这需要用户自行使用下列公式把模和相位角转换成复数的实部和虚部：
> $$
> real = \lvert modulus \rvert \cdot cos(angle)
> $$
> $$
> imag = \lvert modulus \rvert \cdot sin(angle)
> $$

## Numpy

numpy 没有提供以模(modulus)和相位角(angle)直接创建复数tensor的API，需要用户通过上述公式取得实部和虚部后，使用以下方法取得复数array：
```python
np.array(real) + np.array(imag)*j
```

# 实现方法
由于tensorflow和numpy没有实现polar的功能，下面只说明pytorch的实现。

## Pytorch

PyTorch 中实现 polar 的C++代码

 1. [complex.h](https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h#L600)：

```c++
// Thrust does not have complex --> complex version of thrust::proj,
// so this function is not implemented at c10 right now.
// TODO(@zasdfgbnm): implement it by ourselves

// There is no c10 version of std::polar, because std::polar always
// returns std::complex. Use c10::polar instead;

} // namespace std

namespace c10 {

template <typename T>
C10_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T()) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<complex<T>>(thrust::polar(r, theta));
#else
  // std::polar() requires r >= 0, so spell out the explicit implementation to
  // avoid a branch.
  return complex<T>(r * std::cos(theta), r * std::sin(theta));
#endif
}
```

 2. [ComplexKernel.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/ComplexKernel.cpp#L18)
  ```c++
void polar_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "polar_cpu", [&]() {
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
      return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
    });
  });
}
REGISTER_DISPATCH(polar_stub, &polar_kernel);
 ```

 3. [TensorFactories.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L233)
  ```c++
Tensor& polar_out(const Tensor& abs, const Tensor& angle, Tensor& result) {
  complex_check_dtype(result, abs, angle);
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(abs)
      .add_input(angle)
      .check_all_same_dtype(false)
      .build();
  polar_stub(iter.device_type(), iter);
  return result;
}

Tensor polar(const Tensor& abs, const Tensor& angle) {
  complex_check_floating(abs, angle);
  c10::TensorOptions options = abs.options();
  options = options.dtype(toComplexType(abs.scalar_type()));
  Tensor result = at::empty(0, options);
  return at::polar_out(result, abs, angle);
}
 ```


# 四、对比分析
Pytorch的设计思路就是第3部分提到的转换公式。


# 五、设计思路与实现方案

## 命名与参数设计

`paddle.polar(rho, theta, name=None)` 参数说明如下：

- **rho** (Tensor) – 模的张量，数据类型为float32或float64。

- **theta** (Tensor) – 相位角的张量；**theta** 的形状应与 **rho** 的形状相匹配，且数据类型相同。

`Tensor.polar(rho, theta, name=None)` 参数说明如下：

- **rho** (Tensor) – 模的张量，数据类型为float32或float64。

- **theta** (Tensor) – 相位角的张量；**theta** 的形状应与 **rho** 的形状相匹配，且数据类型相同。

输出是一个Tensor，数据类型是 complex64 或者 complex128，与 rho 和 theta 的数值精度一致。

## 底层 OP 设计

主要使用 paddle.complex, paddle.sin, paddle.cos 现有 API 进行设计。

## API 实现方案
- 确保 `rho` 和 `theta` 不都为空，数据类型需为float32或float64
- 确保输入 `rho` 和 `theta` 的数据类型和维度相同
- 确保输入 `rho` 的取值大于或等与0

# 六、测试和验收的考量

1. 结果正确性:

   - 前向计算: `paddle.polar`(和 `Tensor.polar`) 计算结果与 `torch.polar` 计算结果一致。
   - 反向计算:由 Python 组合新增 API 无需验证反向计算。

2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

3. 异常测试:

   - 数据类型检验:
     - rho, theta 要求为 paddle.Tensor
     - rho为非负
   - 具体数值检验:
     - 若rho不是非负，抛出异常
     - 若rho与theta数据类型或维度不相同，抛出异常

# 七、可行性分析和排期规划

方案主要依赖现有 paddle api 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料
[Pytorch 官方文档](https://pytorch.org/docs/stable/generated/torch.polar.html)

# paddle.sgn 设计文档

| API名称                                                    | paddle.sgn                                     | 
|----------------------------------------------------------|------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | TreeML                                         | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-07-05                                     | 
| 版本号                                                      | V1.0                                           | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                        | 
| 文件名                                                      | 20220705_api_design_for_sgn.md<br> | 

# 一、概述

## 1、相关背景

对于复数张量，此函数返回一个新的张量，其元素与 input 元素的角度相同且绝对值为1；
对于非复数张量，此函数返回 input 元素的符号。
此任务的目标是在 Paddle 框架中，新增 sgn API，调用路径为：paddle.sgn 和 Tensor.sgn。


## 3、意义

完善paddle中对于复数的sgn运算。

# 二、飞桨现状

目前paddle拥有类似的对于实数进行运算的API：sign，
sign对输入x中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零，
sgn是对sign复数功能的实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.sgn(input, *, out=None)` ， 支持复数的符号函数运算：

 ```
 This function is an extension of torch.sign() to complex tensors. It computes a new tensor whose elements have the same
 angles as the corresponding elements of input and absolute values (i.e. magnitudes) of one for complex tensors and is
 equivalent to torch.sign() for non-complex tensors.
 ```
官方文档链接为：https://pytorch.org/docs/stable/generated/torch.sgn.html?highlight=sgn#torch.sgn

## Tensorflow

在Tensorflow中sign此API同时支持复数与实数的符号函数运算：
 ```
y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
对于复数，y = sign(x) = x / |x| if x != 0, otherwise y = 0.
 ```
官方文档链接为：https://www.tensorflow.org/api_docs/python/tf/math/sign

## Numpy

在Numpy中sign此API同时支持复数与实数的符号函数运算，但其复数运算所得到的结果为sign(x.real) + 0j：
 ```
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
For complex inputs, the sign function returns sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j.
complex(nan, 0) is returned for complex nan inputs.
There is more than one definition of sign in common use for complex numbers. The definition used here is equivalent to
 x/sqrt(x*x) which is different from x/|x|a common alternative, .
 ```
官方文档链接为：https://numpy.org/doc/stable/reference/generated/numpy.sign.html?highlight=sign#numpy.sign

### 实现方法

代码如下：

Pytorch中使用C++来实现复数功能

 ```
 template<typename T>
inline c10::complex<T> sgn_impl (c10::complex<T> z) {
  if (z == c10::complex<T>(0, 0)) {
    return c10::complex<T>(0, 0);
  } else {
    return z / zabs(z);
  }
}

 ```
github链接为：https://github.com/pytorch/pytorch/blob/d7fc864f0da461512fb7b972f04e24e296bd266d/aten/src/ATen/native/cpu/zmath.h#L156-L163
Tensorflow中使用python实现复数功能

```
  if x.dtype.is_complex:
    return gen_math_ops.div_no_nan(
        x,
        cast(
            gen_math_ops.complex_abs(
                x,
                Tout=dtypes.float32
                if x.dtype == dtypes.complex64 else dtypes.float64),
            dtype=x.dtype),
        name=name)
  return gen_math_ops.sign(x, name=name)
```
github链接为：https://github.com/tensorflow/tensorflow/blob/7272e9f1f52ffe1b5aee67d1af3c2127634ab47d/tensorflow/python/ops/math_ops.py#L746-L790

# 四、对比分析

Tensorflow与Pytorch对于实现复数功能部分的代码核心逻辑相同，torch的代码使用C++实现但它将实数和复数拆分成了两个API，类似于paddle的想法；
Tensorflow的代码使用Python实现但它将两个功能合于一个API中。
鉴于两段代码的逻辑类似，故参考Pytorch的代码或参考Tensorflow的代码皆可。

# 五、方案设计

## 命名与参数设计

API设计为`paddle.sgn(x, name=None)`和`paddle.Tensor.sgn(x, name=None)`
命名与参数顺序为：形参名`input`->`x`,  与paddle其他API保持一致性，不影响实际功能使用。


## 底层OP设计

使用已有API进行组合，不再单独设计底层OP。


## API实现方案

使用is_complex判断输入是否为复数、若为实数则使用sign进行运算；若为复数则使用as_real将其转化为实数tensor，将其中的非零部分除以它自己的绝对值
，最后再使用as_complex将其转换回复数返回。

# 六、测试和验收的考量

测试考虑的case如下：

- 编程范式场景：覆盖静态图和动态图测试场景
- 硬件场景：覆盖CPU和GPU测试场景
- Tensor精度场景：支持float16， float32 ， float64， complex64 , complex128
- 参数组合场景
- 计算精度：前向计算，和numpy实现的函数对比结果；反向计算，由Python组合的新增API无需验证反向计算
- 异常测试：由于使用了已有API：sign，该API不支持整型运算，仅支持float16， float32 或 float64，所以需要做数据类型的异常测试
  


# 七、可行性分析及规划排期

方案主要依赖paddle现有API组合而成，并自行实现核心算法。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无
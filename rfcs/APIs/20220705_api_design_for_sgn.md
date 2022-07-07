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

对于复数张量，此函数返回一个新的张量，其元素与 input 元素的角度相同且绝对值为 1。对于非复数张量，此函数返回 input 元素的符号。此任务的目标是
在 Paddle 框架中，新增 sgn API，调用路径为：paddle.sgn 和 Tensor.sgn。


## 3、意义

完善paddle中对于复数的sgn运算

# 二、飞桨现状

目前paddle拥有类似的对于实数进行运算的API：sign
sign对输入x中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零。
sgn是对sign复数功能的实现

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.sgn(input, *, out=None)` ， 在pytorch中，介绍为：

 ```
 This function is an extension of torch.sign() to complex tensors. It computes a new tensor whose elements have the same
 angles as the corresponding elements of input and absolute values (i.e. magnitudes) of one for complex tensors and is
 equivalent to torch.sign() for non-complex tensors.
 ```

可以支持complex运算

## Tensorflow

在Tensorflow中sign此API同时支持复数与实数运算：
 ```
y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
对于复数，y = sign(x) = x / |x| if x != 0, otherwise y = 0.
 ```
## Numpy

在Numpy中无专门对于复数计算的符号函数，其拥有相关API，sign：
 ```
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
For complex inputs, the sign function returns sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j.
complex(nan, 0) is returned for complex nan inputs.
 ```
其中对于复数的返回并不是我们期望得到的

### 实现方法

代码如下

torch中使用C++来实现类似功能

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

# 四、对比分析

只有pytorch和paddle类似拆分为两个API分别实现实数和复数功能的符号函数运算，且该运算实现的数学逻辑简单，故参考pytorch的代码

# 五、方案设计

## 命名与参数设计

API设计为`paddle.sgn(x, name=None)`和`paddle.Tensor.sgn(x, name=None)`
命名与参数顺序为：形参名`input`->`x`,  与paddle其他API保持一致性，不影响实际功能使用。


## 底层OP设计

使用已有API进行组合，不再单独设计底层OP。
具体使用了：sign,abs,is_complex,as_real,reshape,as_complex
按照计算逻辑组合API实现复数功能
y = sign(x) = x / |x| if x != 0, otherwise y = 0

## API实现方案

如果复数为0，则直接返回0；否则，返回该复数除以它的绝对值的值
对于非复数直接返回其符号

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性
- 反向
- 异常测试：由于使用了已有API：sign 该API不支持整型运算，仅支持float16， float32 或 float64，所以需要做数据类型的异常测试
  


# 七、可行性分析及规划排期

方案主要依赖paddle现有API组合而成，并自行实现核心算法

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
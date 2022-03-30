# logcumsumexp 设计文档

| API名称                                                      | 新增API名称                                 |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | paddle.logcumsumexp 和 Tensor.logcumsumexp  |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-30                                  |
| 版本号                                                       | V1.0                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                     |
| 文件名                                                       | 20220330_api_design_for_logcumsumexp.md<br> |


# 一、概述

## 1、相关背景

logcumsumexp(x) 会计算 x 沿某一个坐标轴的以 e 为底的指数的前缀和（prefix scan）的自然对数，即 log(cumsum(exp(x)))。

## 2、功能目标

在飞桨中增加 logcumsumexp API。

## 3、意义

飞桨将支持 logcumsumexp API。

# 二、飞桨现状

在飞桨中，logcumsumexp 可以通过已有的 API 组合而成：paddle.log(paddle.cumsum(paddle.exp(x)))，但这样做的数值稳定性很差。


# 三、业内方案调研

PyTorch：API 为 `torch.logcumsumexp(input, dim, *, out=None)` ；

TensorFlow：API 为 `tf.math.cumulative_logsumexp(x, axis=0, exclusive=False, reverse=False, name=None)` ，在 TensorFlow Probability 库内有类似的 `tfp.math.log_cumsum_exp(x, axis=-1, name=None) `；

NumPy：没有实现该操作。

PyTorch 和 TensorFlow 的实现方式是类似的（可以参阅[这里](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp#L128)），为了达到较好的数值稳定性，都采用如下的方法：

基于 log_add_exp 这个满足结合律的运算进行 prefix scan：

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20log%5C_add%5C_exp%28x_1%2C%20x_2%29%20%3D%20log%28exp%28x_1%29%20&plus;%20exp%28x_2%29%29)

假设 prefix scan 类型为 inclusive prefix scan，则运算结果为

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20y_1%20%3D%20x_1)

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20y_2%20%3D%20log%5C_add%5C_exp%28y_1%2C%20x_2%29%20%3D%20log%28exp%28x_1%29%20&plus;%20exp%28x_2%29%29)

![img](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20y_3%20%26%3D%20log%5C_add%5C_exp%28y_2%2C%20x_3%29%20%5C%5C%20%26%3D%20log%28exp%28log%28exp%28x_1%29%20&plus;%20exp%28x_2%29%29%29%20&plus;%20exp%28x_3%29%29%20%5C%5C%20%26%3D%20log%28exp%28x_1%29%20&plus;%20exp%28x_2%29%20&plus;%20exp%28x_3%29%29%20%5Cend%7Balign*%7D)

等等。prefix scan 有成熟的并行算法。

log_add_exp 这个二元操作可以以这样的方法运算以避免指数引起的溢出：

![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20log%5C_add%5C_exp%28x%2C%20y%29%20%26%3D%20log%28exp%28x%29%20&plus;%20exp%28y%29%29%20%5C%5C%20%26%3D%20log%281%20&plus;%20exp%28min%28x%2C%20y%29%20-%20max%28x%2C%20y%29%29%29%20&plus;%20max%28x%2C%20y%29%20%5Cend%7Balign*%7D)



# 四、对比分析

PyTorch 和 TensorFlow 的实现思路是一样的，但 TensorFlow 支持更丰富的功能：exclusive/inclusive 和是否 reverse。

# 五、设计思路与实现方案

## 命名与参数设计

对齐 TensorFlow 所支持的参数（即支持 exclusive 和 reverse，这也和 Paddle 自己的 cumsum op 相一致），同时参考 python/paddle/tensor/math.py 中的 cumsum op API，支持 dtype 参数。即提供两个 API：

```python
paddle.logcumsumexp(x, axis=None, exclusive=False, reverse=False, dtype=None, name=None)
Tensor.logcumsumexp(axis=None, exclusive=False, reverse=False, dtype=None, name=None)

```

axis 默认为 None，表示将 tensor flatten 后再进行操作；dtype 会用于将 input tensor cast 到另一种数据类型，防止累加的溢出；exclusive 表示是否进行 exclusive prefix scan；reverse 表示是否是从尾部开始由后往前计算累加和。

## 底层OP设计

一定程度上可以仿照 Paddle 中已有的 cumsum op 来实现。Paddle 的 cumsum CUDA kernel 中，直接调用了 thrust 的 API `thrust::exclusive_scan` 或 `thrust::inclusive_scan` 。幸运的是 thrust 有一族 [Transformed Prefix Scans 接口](https://thrust.github.io/doc/group__transformed__prefixsums.html)，支持以任意一个满足结合律的二元运算代替 “加法” 的角色。因此 logcumsumexp 的 CUDA kernel 只要使用这族接口，将 log_add_exp 作为参与 prefix scan 的二元运算即可。

但 Paddle cumsum CPU kernel 使用 eigen 作为运算库，eigen 只有普通的 cumsum 和 cumprod 接口，而不支持任意满足结合律的二元运算，因此 Paddle logcumsumexp 的 CPU Kernel 可能只能先写一个串行的 naive 版而不容易实现高效的并行。

## API实现方案

仍是类似于 Paddle cumsum op 即可。

# 六、测试和验收的考量

实现基于 NumPy 的参考实现，将 Paddle logcumsumexp kernel 的结果与参考实现对比。在 CPU 和 GPU、静态图与动态图模式，以及各种参数设置下，均与 NumPy 的结果一致。

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料

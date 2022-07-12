# paddle.Tensor.remainder_ 设计文档

|API名称 | paddle.remainder_                         | 
|---|-------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | thunder95                                 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-08                                | 
|版本号 | V1.0                                      | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   | 
|文件名 | 20220708_api_design_for_remainder_.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度， Paddle需要支持API`paddle.remainder_`的功能。
## 2、功能目标
在现有的Paddle API`paddle.Tensor.remainder`基础上实现其inplace版本。
## 3、意义
飞桨支持`paddle.remainder_`的API功能。

# 二、飞桨现状
目前Paddle已经支持`paddle.Tensor.remainder`相关的API功能，[代码位置](https://github.com/PaddlePaddle/Paddle/blob/5369378ba74efb99b44c245fd8097b159615234f/python/paddle/tensor/math.py#L704) 。
可以在此API基础上做一些扩展注册支持inplace操作。

# 三、业内方案调研
## Numpy 
### 实现方法
Numpy目前仅支持remainder, 并不支持remainder_，[remainder文档](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html)　。
使用示例：
```Python
    import numpy as np
    np.remainder(np.arange(7), 5)
```
Numpy中的计算原理是元素级的求余运算，支持带条件的广播机制。

## Pytorch
Pytorch中有API`Tensor.remainder_(divisor) → Tensor`, 是torch.remainder的inplace版本，在pytorch中，介绍为：
```
Computes Python’s modulus operation entrywise. The result has the same sign as the divisor other and its absolute value is less than that of other.
```

### 实现方法
计算方式按照python的取模运算，　只支持整数和浮点数计算，不支持复数，使用示例：
```Python
    import torch
    torch.remainder(torch.tensor([1, 2, 3, 4, 5]), -1.5)
```

## Tensorflow

TensorFlow实验性引入了对 NumPy API 子集的支持。可借此模块，运行由 TensorFlow 加速的 NumPy 代码。

```python
    tf.experimental.numpy.remainder(
        x1, x2
    )
```
但不支持Numpy中out, where, casting, order, dtype, subok, signature, extobj等变量。


# 四、对比分析
该算子实现比较简单，业内各个方案的差异性都不大，都支持基本的广播机制，Numpy中虽不支持inplace版本，但能支持更多参数变量。

# 五、方案设计
## 命名与参数设计
API设计和参数跟现在非inplace版本的remainder算子保持一致。
输入Ｔensor: `x`和`y`

## 底层OP设计
基于飞桨现有的remainder算子，　底层使用elementwise_mod_op元素取模运算，不单独设计OP。

## API实现方案
参考`subtract_`实现方式，

- 参考paddle/fluid/operators/elementwise/elementwise_sub_op.cc，完成elementwise_mod_op的inplace版本注册。
- 在 Paddle repo 的 python/paddle/tensor/math.py 文件实现remainder_算子接口.
- 在 Paddle repo 的 python/paddle/fluid/tests/unittests/test_inplace.py 和 python/paddle/fluid/tests/unittests/test_elementwise_mod_op.py中，分别加上 remainder_ 的测试代码。

# 六、测试和验收的考量
测试考虑的case如下：

- 测试用例与paddle.Tensor.remainder保持一致，这里没有什么改变；
- 测试inplace版本和非inplace版本输出结果保持一致；

# 七、可行性分析及规划排期

`paddle.remainder`已经在 Paddle repo 的 python/paddle/tensor/math.py [目录中](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py)　。
只需要在该算子基础上实现inplace版本的接口，整个过程可快速完成实现。

# 八、影响面
对原有算子增加inplace版本的注册，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无


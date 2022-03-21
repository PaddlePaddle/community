# paddle.nn.softmax2D设计文档

| API名称                                                      | paddle.nn.softmax2D                         |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                             |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-18                                  |
| 版本号                                                       | V1.1                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                     |
| 文件名                                                       | 20220318_api_design_for_nn_Softmax2D.md<br> |



# 一、概述

## 1、相关背景

paddle.nn.Softmax2D 是 [paddle.nn.Softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax_cn.html#softmax) 的变体，其针对 3D 或者 4D 的 tensor 在空间维度计算softmax，从而输出 tensor 在每个空间维度（channels, hj, wj）的 tensor 求和为1。

## 2、功能目标

在 paddle 框架中新增 Softmax2D API，调用路径为：paddle.nn.Softmax2D。

## 3、意义

飞桨支持Softmax2D组网API。

# 二、飞桨现状

paddle中有相关API [paddle.nn.softmax](https://github.com/PaddlePaddle/Paddle/blob/release/2.2/python/paddle/nn/layer/activation.py#L1051)和[paddle.nn.functional.softmax](https://github.com/PaddlePaddle/Paddle/blob/0ee230a7d3177f791d2a5388ab4dffdccc03f4aa/python/paddle/nn/functional/activation.py#L790)，调用底层算子实现，`softmax2d`可以使用`paddle.nn.functional.softmax`API进行实现。

`softmax`的整体逻辑如下：

1. 检查输入的数据类型是否为float；

2. 若指定了`dtype`，则先将输入转换为该数据类型；

3. 将输入指定的维度置换到最后一维；

4. 将置换后的结果变换为二维矩阵，二维矩阵第一维（列的长度）是输入除最后一维之外的其他维度值的乘积，第二维（行长度）和输入 `axis` 维的长度相同；对于矩阵的每一行，softmax操作对其进行重新缩放，使得该行的每个元素在 [0,1] 范围内，并且总和为1；

5. 对每一行都计算softmax的结果，计算公式如下：
   $$
   Softmax[i,j] = \frac{\exp(x[i,j])}{\sum_j(exp(x[i,j])}
   $$
   对于第i行上的第j个值，其计算结果为该值的指数值比上第i行上所有值的指数值之和；

6. 执行3、4步的逆步骤，将结果恢复到输入的维度。

`softmax2d`实际上就是在空间维度计算softmax，从而使输出 tensor 在每个空间维度$(channels, h_j, w_j)$的 tensor 求和为1，即表示空间位置上的某个通道向量求和为1。

由上述第4步描述可知，将`axis`参数固定为-3， 即在channels的维度上进行计算即可，在实际运算中，会将输入的形状`reshape`为$height*width, channels$或$batch*height*width, channels$，这样一来每一行的结果求和都为1，便符合了Softmax2D的需求。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.nn.Softmax2d()`。

在Pytorch中，介绍为：

```
Applies SoftMax over features to each spatial location.

When given an image of Channels x Height x Width, 
it will apply Softmax to each location (Channels, h_i, w_j)
```

### 实现方法

在实现方法上,  Pytorch直接调用`torch.nn.functional.softmax`实现，[代码位置](https://github.com/pytorch/pytorch/blob/727e24313b23f5f5fd0b139bb3d3158f1dcc4d1f/torch/nn/modules/activation.py#L1292)。

核心代码为：

```python
def forward(self, input: Tensor) -> Tensor:
	assert input.dim() == 4 or input.dim(
	) == 3, 'Softmax2d requires a 3D or 4D tensor as input'
	return F.softmax(input, -3, _stacklevel=5)
```

主要逻辑如下：

1. 对输入的维度进行判断；
2. 直接调用`torch.functional.softmax`，并将`dim`固定为-3。

# 四、对比分析

并未找到其他实现。

# 五、设计思路与实现方案

## 命名与参数设计

API 设计为 `paddle.nn.Softmax2D()`。

## 底层OP设计

直接调用现有API，无需设计底层OP。

## API实现方案

- 判断输入的维度，若不是3维或4维，则抛出错误；
- 固定`paddle.nn.functional.softmax`中的`axis`参数为-3，封装在`paddle.nn.Softmax2D`中；
- 返回上述`softmax`的结果即可。

# 六、测试和验收的考量

- 在静态图、动态图下，与numpy结果的一致性；
- 在GPU、CPU上，与numpy结果的一致性；
- 错误检查：当输入Tensor不是3维或4维时能正确抛出错误。

# 七、可行性分析和排期规划

方案直接依赖现有API`paddle.nn.functional.softmax`完成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 九、评审意见



# 名词解释

无

# 附件及参考资料

无
# paddle.nn.softmax2d设计文档

| API名称                                                      | 新增API名称                                 |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                             |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-18                                  |
| 版本号                                                       | 此设计文档的版本号，如V1.0                  |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                     |
| 文件名                                                       | 20220318_api_design_for_nn_Softmax2D.md<br> |



# 一、概述

## 1、相关背景

paddle.nn.Softmax2D 是 [paddle.nn.Softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax_cn.html#softmax) 的变体，其针对 3D 或者 4D 的 tensor 在空间维度计算softmax，从而输出 tensor 在每个空间维度（channels, hj, wj）的 tensor 求和为1。

## 2、功能目标

在 paddle 框架中新增 Softmax2D API，调用路径为：paddle.nn.Softmax2D。

## 3、意义

飞浆支持Softmax2D组网API

# 二、飞桨现状

可以使用`Softmax`API进行实现

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

在实现方法上,  Pytorch直接调用`torch.nn.functional.softmax`实现，[代码位置](hhttps://github.com/pytorch/pytorch/blob/727e24313b23f5f5fd0b139bb3d3158f1dcc4d1f/torch/nn/modules/activation.py#L1292)。

核心代码为：

```python
def forward(self, input: Tensor) -> Tensor:
	assert input.dim() == 4 or input.dim(
	) == 3, 'Softmax2d requires a 3D or 4D tensor as input'
	return F.softmax(input, -3, _stacklevel=5)
```



# 四、对比分析



# 五、设计思路与实现方案

## 命名与参数设计

API 设计为 `paddle.nn.Softmax2D()`。

## 底层OP设计

直接调用现有API，无需设计底层OP。

## API实现方案

固定`paddle.nn.functional`中的`axis`为-3，封装在`paddle.nn.Softmax2D`中即可。

# 六、测试和验收的考量

- 在静态图、动态图下，与numpy结果的一致性；
- 错误检查：当输入Tensor不是3维或4维时能正确抛出错误。

# 七、可行性分析和排期规划

方案直接依赖现有API，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 九、评审意见

（由评审人填写，开发者无需填写）

| 问题 | 提出人 | 处理说明 | 状态 |
| ---- | ------ | -------- | ---- |
|      |        |          |      |

# 名词解释

无

# 附件及参考资料

无
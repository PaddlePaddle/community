# paddle.index_fill 设计文档

| API名称                                                      | paddle.index_fill |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者   | robinbg                  |
| 提交时间| 2023-09-16                |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名                                                       | 20230916_api_design_for_index_fill.md |

# 一、概述

## 1、相关背景

对于 nd tensor, 沿着某个轴 axis 取 (n-1)d 的切片，索引位置是 index, 并且将 value 中值填充到这些切片上。其中 value 是一个 scalar 或者 0d tensor, 该运算需要支持微分。

## 2、功能目标

index_fill API 是一个按轴和索引填充值到目标张量的API。此任务的目标是在 Paddle 框架中，新增 index_fill API，同时实现inplace和非inplace版本，调用路径为：

- paddle.index_fill 作为独立的函数调用，非 inplace
- paddle.index_fill_，作为独立的函数，inplace 地修改输入；
- Tensor.index_fill， 作为 Tensor 的方法使用，非 inplace;
- Tensor.index_fill_，作为 Tensor 的方法使用， inplace 修改输入；

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中已实现了inplace版本torch.Tensor.index_fill_和非inplace版本torch.Tensor.index_fill的API

### 实现方法
调用和底层C++ API。


## MindSpore
MindSpore中实现了非inplace版本mindspore.Tensor.index_fill和非inplace版本mindspore.ops.index_fill


### 实现方法
调用和底层C++ API。



# 四、对比分析

可以直接参考的实现是pytorch，但是鉴于paddle中已有index_put API，可以想到组合index_put和其它Paddle API，在python端实现index_fill的功能，由此利用index_put已经实现的动静图、前反向功能



# 五、方案设计

## 命名与参数设计

新增API设计为:

`paddle.index_fill(x, axis, index, fill_value)`

`paddle.index_fill_(x, axis, index, fill_value)`

`Tensor.index_fill(axis, index, fill_value)`

`Tensor.index_fill_(axis, index, fill_value)`



index_fill_支持inplace方式修改输入张量。

axis是index索引选择的轴, 支持int以及0维的Tensor参数类型。

index在指定轴上含索引下标的list of int, tuple of int 或者 1-D Tensor。

fill_value是待填充的数据，参数类型支持bool, int, float以及0维的Tensor。

## 底层OP设计

python端API组合实现

## API实现方案
在 python/paddle/tensor/manipulation.py 中增加index_fill以及index_fill_函数，计算正确的stride之后，参考index_select和index_put算子进行逻辑修改,

在指定轴上指定索引的输入元素梯度为0.0，其他未被选中的元素梯度是1.0,

若fill_value是0维的Tensor，其反向传播的梯度是对应选中的输出梯度的总和sum。

# 六、测试和验收的考量

测试考虑的case如下：

- 和numpy结果的数值的一致性, `paddle.index_fill`和numpy切片操作结果是否一致；

- 参数`axis`校验参数类型int以及0维的tensor，判断axis合法，并进行边界检查；

- 校验参数`index`的正确性，索引边界检查，输出结果的正确性；

- 校验参数fill_value的正确性， 是否是支持的数据类型，当fill_value是0维tensor时梯度正确回传

- 测试在进行反向梯度计算时结果的正确性；

- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

第一阶段：Python算子代码编写；

第二阶段：单元测试；

第三阶段：中英文API文档编写。

# 八、影响面

为独立新增API，对其他模块没有影响

# paddle.incubate.sparse.is_same_shap 设计文档

|API名称 | paddle.incubate.sparse.is_same_shap       | 
|---|-------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | PeachML                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-12                                | 
|版本号 | V1.0                                      | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   | 
|文件名 | 20220712_api_design_for_is_same_shape.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，is_same_shape 是一个基础的形状比较操作，目前 Paddle 中还没有 is_same_shape 算子。 
本任务的目标是在 Paddle 中添加 paddle.incubate.sparse.is_same_shap 算子， 实现输入是 dense、coo、csr 之间的形状比较。 
Paddle需要扩充API,新增 is_same_shape API， 调用路径为：`paddle.incubate.sparse.is_same_shap`

## 3、意义

支持稀疏tensor之间和稀疏tensor与稠密tensor之间的形状比较，打通Sparse与Dense领域的操作。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch无此API

## Numpy
Numpy中一般使用`a.shape()==b.shape()`进行判断

# 四、对比分析

paddle中要实现九种交叉比较，故自行实现

# 五、方案设计

## 命名与参数设计

在 python/paddle/incubate/sparse/multiary.py 中新增api，


```python
def is_same_shape(x, y)
```


## 底层OP设计

新增一个tensor method，python端调用

## API实现方案

在 paddle/fluid/pybind/eager_method.cc 中实现，作为TensorObject的类成员函数，对比self.tensor.shape是否一致
静态图实现：目前sparse系列暂不支持静态图

```c
static PyObject* tensor_method_is_same_shape(TensorObject* self,
                                             TensorObject* other,
                                             PyObject* args,
                                             PyObject* kwargs)
```

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性
- 不同类型tensor比较，dense、coo、csr，两两比较，共九种情况

# 七、可行性分析及规划排期

经论证可行

# 八、影响面

为独立新增api，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无

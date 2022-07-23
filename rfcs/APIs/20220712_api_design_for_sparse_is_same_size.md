# paddle.incubate.sparse.is_same_size 设计文档

|API名称 | paddle.incubate.sparse.is_same_size       | 
|---|-------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | PeachML                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-12                                | 
|版本号 | V1.0                                      | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   | 
|文件名 | 20220712_api_design_for_sparse_is_same_size.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，is_same_size 是一个基础的形状比较操作，目前 Paddle 中还没有 sparse 的is_same_size算子。 
本任务的目标是在 Paddle 中添加 sparse.is_same_size 算子， 实现输入是coo 与 dense、csr 与 dense、coo 与 coo、csr 与 csr 四种情况下的形状比较。 
Paddle需要扩充API,新增 sparse.is_same_size API， 调用路径为：`paddle.incubate.sparse.is_same_size`

## 3、意义

支持稀疏tensor之间和稀疏tensor与稠密tensor之间的形状比较，打通Sparse与Dense领域的操作。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.is_same_size` ， 在pytorch中参数列表为`is_same_size(input: Tensor, other: Tensor) -> _bool: ...`

### 实现方法
该api在torch\_C\_VariableFunctions.pyi中出现，在 aten/src/ATen/native/TensorProperties.cpp 中实现，
```c++
bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sizes().equals(other.sizes());
}
```

## Numpy
Numpy中一般使用`a.size()==b.size()`进行判断

# 四、对比分析

paddle中要实现四种交叉比较，故自行实现

# 五、方案设计

## 命名与参数设计

在python/paddle/incubate/sparse/binary.py中新增api

```python
def is_same_size(x, y)
```


## 底层OP设计

可以使用python api组合，无需新增op

## API实现方案

Python端参考numpy实现

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性
- 不同类型tensor比较

# 七、可行性分析及规划排期

方案主要依赖paddle现有api组合而成

# 八、影响面

为独立新增api，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无

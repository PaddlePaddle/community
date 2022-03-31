# paddle.sparse.sqrt 设计文档

|API名称 | paddle.sparse.sqrt                         | 
|--------------------------------------------|--------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | PeachML                                    | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-29                                 | 
|版本号 | V1.0                                       | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                    | 
|文件名 | 20220330_api_design_for_sparse_sqrt.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，sqrt 是一个基础平方根运算操作，目前 Paddle 中还没有 sparse 的sqrt算子。 本任务的目标是在 Paddle 中添加 sparse.sqrt 算子， 实现输入一个 SparseCooTensor
或者 SparseCsrTensor ，输出逐元素平方根的功能。 调用路径为：`paddle.sparse.sqrt`

## 3、意义

支持稀疏tensor求平方根，提高空间利用效率，提升稀疏tensor的计算效率。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.sqrt(input, *, out=None) ` ， 在pytorch中，介绍为：

  ```
  Returns a new tensor with the square-root of the elements of input
  ```

可以支持sparse tensor的平方根

# 四、对比分析

torch设计结构复杂，为了适配paddle phi库的设计模式，可以使用phi中自带的`SqrtFunctor`进行逐元素求值来实现，复杂度为O(n)。

# 五、方案设计

## 命名与参数设计

在paddle/phi/kernels/sparse/目录下， API设计为

  ```    
  void SqrtCooKernel(const Context& dev_ctx,
  const SparseCooTensor& input,
  SparseCooTensor* out);
  ```

和

  ```    
  void SqrtCsrKernel(const Context& dev_ctx,
  const SparseCsrTensor& input,
  SparseCsrTensor* out);
  ```

## 底层OP设计

使用已有op组合实现，主要涉及`SparseCooToCsrKernel`,`SqrtFunctor`和`SparseCsrToCooKernel`。

## API实现方案

主要参考scipy实现，将coo转换成csr再进行乘法，然后转换回coo

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性

# 七、可行性分析及规划排期

方案主要依赖paddle现有op组合而成

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
# paddle.incubate.sparse.slice 设计文档

|API名称 | paddle.incubate.sparse.slice | 
|---|---|
|提交作者 | OccupyMars2025 | 
|提交时间 | 2022-09-12 | 
|版本号   | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20220912_api_design_for_sparse_slice.md | 

# 一、概述

## 1、相关背景

为了提升飞桨稀疏 API 丰富度， Paddle 需要增加 sparse slice 算子。

## 2、功能目标

该OP沿多个轴生成 sparse tensor 的切片。

## 3、意义

提升飞桨稀疏 API 丰富度

# 二、飞桨现状
已有针对 dense tensor 的 [paddle.slice](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/slice_cn.html)，但缺少针对 spares tensor 的相关算子。

# 三、业内方案调研

## Pytorch
无相关实现

可能的问题：
https://discuss.pytorch.org/t/column-row-slicing-a-torch-sparse-tensor/19130
https://stackoverflow.com/questions/50666440/column-row-slicing-a-torch-sparse-tensor
https://exchangetuts.com/columnrow-slicing-a-torch-sparse-tensor-1640244724407864



## TensorFlow

[tf.sparse.slice](https://www.tensorflow.org/api_docs/python/tf/sparse/slice)

# 四、对比分析


# 五、方案设计

## 命名与参数设计

```python
paddle.incubate.sparse.slice(input, axes, starts, ends)
```
- input （Tensor）- 多维 Tensor，数据类型为 float16， float32，float64，int32，或 int64。
- axes （list|tuple）- 数据类型是 int32。表示进行切片的轴。
- starts （list|tuple|Tensor）- 数据类型是 int32。如果 starts 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 Tensor 。如果 starts 的类型是 Tensor，则是1-D Tensor 。表示在各个轴上切片的起始索引值。
- ends （list|tuple|Tensor）- 数据类型是 int32。如果 ends 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 Tensor 。如果 ends 的类型是 Tensor，则是1-D Tensor 。表示在各个轴上切片的结束索引值。

## 底层 OP 设计


## API 实现方案

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 slice 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 coo 格式的 axis 支持任意维度，csr 格式的 axis 可只支持-1（即按行读取）和 None。

Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/incubate/sparse/unary.py 文件；
C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录中；
单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_slice_op.py 文件；
yaml 文件，前反向分别添加到python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。
中文 API 文档，在 docs repo 的 docs/api/paddle/incubate/sparse 目录。

# 六、测试和验收的考量


# 七、可行性分析及规划排期

可行。

2022年9月15日-9月30日完成


# 八、影响面

为独立新增 API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无

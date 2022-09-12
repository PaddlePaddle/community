# paddle.incubate.sparse.any 、paddle.incubate.sparse.sum 、paddle.incubate.sparse.max 和 paddle.incubate.sparse.min 设计文档

|API名称 | paddle.incubate.sparse.any 和  paddle.incubate.sparse.sum  | 
|---|-------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | OccupyMars2025  | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-09-12    | 
|版本号 | V1.0                                      | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop   | 
|文件名 | 20220912_api_design_for_sparse_any_sum_max_and_min.md<br> | 

# 一、概述

## 1、相关背景

any 和 sum 都是基础的 tensor 操作，目前 Paddle 中还没有针对 sparse tensor 的 any 和 sum 算子。因为这两个算子的内部实现机制相似，所以可以一起实现。（实际上any，sum，max，min都是基础的 tensor 操作，内部实现机制都是相似的，可以互相借鉴）
本任务的目标是在 Paddle 中添加针对 sparse tensor 的 any 和 sum 算子。
Paddle需要扩充 API ,新增 sparse any 和 sparse sum 的 API， 调用路径为：`paddle.incubate.sparse.any` 和 `paddle.incubate.sparse.sum`

## 3、意义

添加针对稀疏tensor的基础tensor操作。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

可以参考[torch.sparse.sum](https://pytorch.org/docs/stable/generated/torch.sparse.sum.html#torch.sparse.sum)
sum算子的python端的代码位于torch\sparse\__init__.py （line 153）
C++端的代码位于aten\src\ATen\native\sparse\SparseTensorMath.cpp (line 1515)

## tensorflow

可以参考 [tf.sparse.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/sparse/reduce_sum) 和 [tf.sparse.reduce_max](https://www.tensorflow.org/api_docs/python/tf/sparse/reduce_max) 的实现

# 四、对比分析

paddle必须要自行实现，要写新的相关的C++ kernel 和 CUDA kernel

# 五、方案设计

## 命名与参数设计

在 python/paddle/incubate/sparse/unary.py 中新增api，


```python
def any(x, axis=None, keepdim=False, name=None)
def sum(x, axis=None, keepdim=False, name=None)
def max(x, axis=None, keepdim=False, name=None)
def min(x, axis=None, keepdim=False, name=None)
```
考虑是否增加 output_is_sparse=False 这个参数


## 底层OP设计

新增几个tensor method，python端调用

## API实现方案

在 paddle/fluid/pybind/eager_method.cc 中实现，作为TensorObject的类成员函数。
静态图实现：目前sparse系列暂不支持静态图

```c
static PyObject* tensor_method_any(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)
static PyObject* tensor_method_sum(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)
static PyObject* tensor_method_max(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)                 
static PyObject* tensor_method_min(TensorObject* self,
                                   PyObject* args,
                                   PyObject* kwargs)     
```

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性
- axis的错误数值能否引发index error

# 七、可行性分析及规划排期

经论证可行

# 八、影响面

为独立新增api，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

本 RFC 参考了https://github.com/PaddlePaddle/community/pull/184，在这里对
https://github.com/gsq7474741 表示感谢。

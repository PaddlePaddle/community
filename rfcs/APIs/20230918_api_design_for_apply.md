
#为 Paddle 新增 apply API 设计文档

| API名称 | 新增API名称 | 
|---|---|
| 提交作者 | robinbg| 
| 提交时间 | 2023-09-18 | 
| 版本号 | V1.0 | 
| 依赖飞桨版本 | develop| 
| 文件名 | 20230918_api_design_for_apply.md | 

# 一、概述

## 1、相关背景

为了增强Paddle框架的功能并与其他框架保持竞争力，我们决定为Tensor添加一个新的API，它允许用户应用Python函数到Tensor的每个元素。

## 2、功能目标

- 为Tensor添加 `apply(callable)` 方法，返回新的Tensor，存放计算结果。
- 为Tensor添加 `apply_(callable)` 方法，inplace修改输入Tensor。

## 3、意义

该功能将提供更大的灵活性，使开发人员能够轻松地应用自定义函数到Tensor的每个元素，从而实现更复杂的操作。

# 二、飞桨现状

Paddle当前不支持此功能。

# 三、业内方案调研

TensorFlow、PyTorch 和 NumPy 都提供了允许用户对数组或张量元素应用函数的功能。

- **TensorFlow**: TensorFlow 提供了 `tf.map_fn` 方法，使用户能够对张量的元素应用函数。
  
- **PyTorch**: PyTorch 通过 `torch.Tensor.apply_` 方法，允许用户在原地应用函数到张量的元素。但需要注意，此方法仅限于CPU张量，并且不鼓励在新代码中使用。

- **NumPy**: NumPy 的 `numpy.vectorize` 函数可以用来为数组的每个元素应用Python函数。

# 四、对比分析

- **TensorFlow**: `tf.map_fn` 方法提供了强大的功能，但其接口相对复杂，不太适合初学者。

- **PyTorch**: 虽然 `torch.Tensor.apply_` 方法提供了所需的功能，但由于其局限性，它在新代码中不被推荐使用。

- **NumPy**: `numpy.vectorize` 是一个简单且强大的工具，但它在某些情况下可能不如专门设计的函数高效。

Paddle 的 `apply` 和 `apply_` 方法的设计目标是结合以上框架的优点，提供一个简单、高效且功能强大的接口。

# 五、设计思路与实现方案

## 命名与参数设计

参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)

API将被命名为 `apply` 和 `apply_`，其中 `apply_` 表示原地操作。

参数:
- `callable`: 用户提供的Python函数，将应用于Tensor的每个元素。

## 底层OP设计

在 C++ 层面，我们需要考虑如何高效地遍历 Tensor 并应用函数。考虑到可能的多线程和多设备支持，底层实现可能会涉及 CUDA。

## API实现方案

1. **Python层**: 在 Python 层，我们将获取用户的 callable，然后将其传递给 C++ 层。
  
2. **C++层**: 在 C++ 层，我们将遍历 Tensor 的每个元素，并使用 Pybind11 调用用户提供的 Python 函数。对于 `apply_` 方法，我们将直接在原Tensor上修改值。对于 `apply` 方法，我们将创建一个新的 Tensor 来存放结果。

3. **CUDA支持**: 如果 Tensor 在 GPU 上，我们将使用 CUDA 核心来并行处理每个元素。

# 六、测试和验收的考量

- 单测代码，位于Paddle repo的 `test/` 目录。
- 在 `paddle/test/legacy_test/test_inpalce.py` 中新增对应的inplace api单测。

# 七、可行性分析和排期规划

参照PyTorch、Tensorflow相关代码完成迁移，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

- Tensor: Paddle中的基本数据结构，用于表示多维数组。
- API: 应用程序编程接口。
- callable: 可调用的Python对象。

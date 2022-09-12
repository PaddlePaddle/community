# paddle.incubate.sparse.nn.Softmax 设计文档

|API名称 | paddle.incubate.sparse.nn.Softmax | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | OccupyMars2025 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-09-12 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220912_api_design_for_paddle_incubate_sparse_nn_Softmax.md<br> | 


# 一、概述
## 1、相关背景
在当前的 Paddle 框架中，`paddle.incubate.sparse.nn.Softmax` 已经实现，本次任务计划为该 API 增加 coo 格式计算逻辑
## 2、功能目标
为 `paddle.incubate.sparse.nn.Softmax` 增加 coo 格式计算逻辑

## 3、意义
进一步完善 paddle 框架中的稀疏算子
# 二、飞桨现状


# 三、业内方案调研
可以同时参考sparse logsoftmax的实现

## PyTorch
[torch.sparse.softmax](https://pytorch.org/docs/stable/generated/torch.sparse.softmax.html)，[torch.sparse.log_softmax](https://pytorch.org/docs/stable/generated/torch.sparse.log_softmax.html)
C++端代码：aten\src\ATen\native\sparse\SoftMax.cpp

## Tensorflow
[tf.sparse.softmax](https://www.tensorflow.org/api_docs/python/tf/sparse/softmax)
代码位置：https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/ops/sparse_ops.py#L2653-L2706

# 四、对比分析

Pytorch 实现了 sparse softmax 和 sparse logsoftmax，更丰富一些，
tensorflow 只实现了 sparse softmax

# 五、设计思路与实现方案

针对 Paddle 的稀疏 Tensor 格式 COO，需要新增 softmax 的计算逻辑，一共需要新增 1个 kernel 的前向与反向，其中参数 axis 可支持任意维度，注意只需新增 coo 格式的逻辑，csr 格式的已经实现，此次无需实现。

## Python端
Python 实现代码，在 Paddle repo 的 python/paddle/incubate/sparse/nn/functional/activation.py 文件和 python/paddle/incubate/sparse/nn/layer/activation.py 文件；


## 底层OP设计
C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录的 softamx_kernel.h/cc/cu 三个文件中，分别补充 coo 的计算 kernel；

## yaml文件配置
前反向分别添加到python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。

## 中文 API 文档
在 docs repo 的 docs/api/paddle/incubate/sparse 目录。


# 六、测试和验收的考量
单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py 文件；

# 七、可行性分析和排期规划
- 可行性分析

可行

- 排期规划

2022年9月15日~9月30日完成开发。

# 八、影响面

无

# 名词解释

无
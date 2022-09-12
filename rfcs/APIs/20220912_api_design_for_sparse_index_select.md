# paddle.incubate.sparse.index_select 设计文档


|API名称 | paddle.incubate.sparse.index_select |
|---|---|
|提交作者 | OccupyMars2025 |
|提交时间 | 2022-09-12 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20220912_api_design_for_sparse_index_select.md |


# 一、概述
## 1、相关背景
Paddle框架中缺少针对 sparse tensor 的 index_select 算子

## 2、功能目标

在Paddle框架中增加 `paddle.incubate.sparse.index_select` 这个API。

## 3、意义

丰富Paddle的稀疏算子

# 二、飞桨现状

飞桨目前没有提供`paddle.incubate.sparse.index_select`这个API，且无法通过API组合的方式间接实现其功能。


# 三、业内方案调研

## PyTorch
[torch.index_select](https://pytorch.org/docs/stable/generated/torch.index_select.html#torch.index_select) 支持 sparse tensor，详见[pytorch官网](https://pytorch.org/docs/stable/sparse.html#other-functions)

可能会遇到的问题：
https://github.com/pytorch/pytorch/issues/54561，
https://github.com/pytorch/pytorch/issues/61788
https://discuss.pytorch.org/t/index-select-for-sparse-tensors-slower-on-gpu-than-cpu/71645

## tensorflow
没有对应的算子，但可以参考 [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather)的实现



# 四、对比分析


# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.incubate.sparse.index_select(input, axis, index)`

参数：
- input (Tensor) – the input tensor.
- dim (int) – the dimension in which we index
- index (IntTensor or LongTensor) – the 1-D tensor containing the indices to index

## 底层OP设计

## API实现方案
针对 Paddle 的两种稀疏 Tensor 存储格式 COO 与 CSR，都需新增 index_select 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

## 提交内容

API 的设计文档，并提 PR 至 community repo 的 rfcs/APIs 目录；
Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/incubate/sparse/binary.py 文件；
C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录中；
单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_index_select_op.py 文件；
yaml 文件，前反向分别添加到python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。
中文 API 文档，在 docs repo 的 docs/api/paddle/incubate/sparse 目录。

# 六、测试和验收的考量

- 输入合法性及有效性检验；

- 对比与Numpy的结果的一致性：

- CPU、GPU测试。

# 七、可行性分析和排期规划

可行

2022年9月15日-9月30日完成

# 八、影响面

是独立API，不会对其他API产生影响。

# 名词解释
无

# 附件及参考资料
无

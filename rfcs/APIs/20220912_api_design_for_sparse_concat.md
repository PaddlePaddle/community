# paddle.incubate.sparse.concat 设计文档

| API名称      | paddle.incubate.sparse.concat       |
| ------------ | ------------------------------------ |
| 提交作者     | OccupyMars2025                      |
| 提交时间     | 2022-09-12                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop                            |
| 文件名       | 20220912_api_design_for_sparse_concat.md |

# 一、概述

## 1、相关背景

为了提升飞桨稀疏API丰富度，Paddle需要扩充API`paddle.incubate.sparse.concat`

## 2、功能目标

沿着给定的 dim，拼接一系列的 sparse tensor

## 3、意义

丰富 paddle 稀疏算子。

# 二、飞桨现状

- 目前paddle缺少相关功能实现

# 三、业内方案调研

## pytorch
[torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) 能够支持 sparse tensor，没有对应的稀疏算子， 详见 [pytorch文档](https://pytorch.org/docs/stable/sparse.html#other-functions)

## tensorflow
[tf.sparse.concat](https://www.tensorflow.org/api_docs/python/tf/sparse/concat), [tensorflow::ops::SparseConcat](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-concat)

## paddle
针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，都需新增 concat 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。


# 五、方案设计

## 命名与参数设计

- 函数名称: paddle.incubate.sparse.concat(x, axis=0, name=None)
- 功能描述: 对输入沿 axis 轴进行联结，返回一个新的 Tensor
- 输入参数
  - x: (list|tuple) 待联结的Tensor list或者Tensor tuple ，支持的数据类型为：bool、float16、float32、float64、int32、int64、uint8， x 中所有Tensor的数据类型应该一致。
  - axis: (int|Tensor，可选) - 指定对输入 x 进行运算的轴，可以是整数或者形状为[1]的Tensor，数据类型为int32或者int64。 axis 的有效范围是[-R, R)，R是输入 x 中Tensor的维度， axis 为负值时与 axis+R 等价。默认值为0。
  - name: (str，可选) 
- 返回值:
  - 拼接后的一个新的tensor

## 底层OP设计


## API实现方案

1. Python 实现代码 & 英文 API 文档，在 Paddle repo 新建 python/paddle/incubate/sparse/multiary.py 文件；
2. C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录中；
3. 单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_concat_op.py 文件；
4. yaml 文件，前反向分别添加到python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。
5. 中文 API 文档，在 docs repo 的 docs/api/paddle/incubate/sparse 目录。

# 六、测试和验收的考量

- 计算精度：
  - 前向计算：通过 numpy 实现的函数的对比结果。
  - 反向计算：

- 异常测试：需对于参数异常值输入，应该有友好的报错信息及异常反馈。
- 参数组合场景：常规覆盖 API 的全部入参，需要对全部入参进行参数有效性和边界值测试，同时可选参数也需有相应的测试覆盖。

# 七、可行性分析及规划排期

可行

2022年9月15日-9月30日完成

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
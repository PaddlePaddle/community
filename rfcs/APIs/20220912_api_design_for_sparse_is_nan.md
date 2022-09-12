# paddle.incubate.sparse.is_nan 设计文档

| API 名称     | paddle.incubate.sparse.is_nan        |
| ------------ | ---------------------------------- |
| 提交作者     | OccupyMars2025       |
| 提交时间     | 2022-09-12                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | develop                            |
| 文件名       | 20220912_api_design_for_sparse_is_nan.md     |

# 一、概述

## 1、相关背景

paddle框架缺少针对 sparse tensor 的 is_nan 算子。

## 2、功能目标

增加 API `paddle.incubate.sparse.is_nan`，返回输入 sparse tensor的每一个值是否为 +/-NaN 。

## 3、意义

丰富 paddle 的稀疏API

# 二、飞桨现状

目前 飞桨没有 API `paddle.incubate.sparse.is_nan`

# 三、业内方案调研

## PyTorch

[torch.isnan](https://pytorch.org/docs/stable/generated/torch.isnan.html#torch.isnan) 支持 sparse coo tensor


## TensorFlow
无相关实现


# 四、对比分析


# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.incubate.sparse.is_nan(x, name=None)
```

## 底层 OP 设计

可以调用 paddle.isnan 的底层 kernel 来辅助实现

## API 实现方案

针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR，需新增 is_nan 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/incubate/sparse/unary.py 文件；
C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录；
单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_is_nan.py 文件；
yaml 文件，前反向分别添加到 python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。
中文 API 文档，在 docs repo 的 docs/api/paddle/incubate/sparse 目录。


# 六、测试和验收的考量



# 七、可行性分析及规划排期

2022年9月15日-9月20日完成

# 八、影响面

增加了一个独立 API，对其他模块无影响

# 名词解释

无

# 附件及参考资料

无
# paddle.cdist 设计文档
|API名称 | paddle.cdist                                 | 
|---|----------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | [yangguohao](https://github.com/yangguohao/) |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-05-19                                   | 
|版本号 | V2.0                                         | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                      | 
|文件名 | 20230519_api_design_for_cdist.md             | 


# 一、概述
## 1、相关背景

Issue: 【PaddlePaddle Hackathon 4】2、为 Paddle 新增 cdist API 
`cdist` 用于计算两组行向量集合中每一对之间的 `p-norm` 距离。

## 2、功能目标

为 Paddle 新增 cdist API。dist API 用于计算两个输入 Tensor 的 p 范数（p-norm），计算结果为形状为 [] 的 0-D Tensor，而 cdist API 则用于计算两个输入 Tensor 的所有行向量对的 p 范数（p-norm），输出结果的形状和两个 Tensor 乘积的形状一致。
## 3、意义
为 Paddle 新增 cdist API，提供距离计算功能。
# 二、飞桨现状
对飞桨框架目前不支持此功能，可用其他API组合实现的此功能。
# 三、业内方案调研

## Pytorch
Pytorch 中使用的 API 格式如下：
`torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')`
在 Pytorch 中， 通过在 c++ 与 cuda 端编写了 op 算子，之后在python端调用。通过参数 p 可以等价于多种距离的计算。 并且在 p=2 的时候可以通过矩阵运算加速运算的过程。
# 四、对比分析
**PyTorch** 的方案中的功能最直观也最全面，使用起来用户友好。
# 五、设计思路与实现方案
## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.cdist(x1, x2, p=2.0)`。其中 `x1` 为 `B1 X ... X Bn X P X M` 张量，`x2` 为 `B1 X ... X Bn X R X M` 张量。`p` 为 p-范数对应的 p 值，p ∈[0,∞]。输出张量的形状为 `B1 X ... X Bn X P X R`。

这里与 `torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')` 的设计不同之处是去除了 `compute_mode` 参数，其作用是使用矩阵乘法加速欧氏距离（p=2）的计算。在 PyTorch 中，参数 `compute_mode='use_mm_for_euclid_dist_if_necessary'`，当 P > 25 或 R > 25 时，则使用矩阵乘法加速计算，否则使用普通的欧氏距离计算。

对于 PaddlePaddle 将这一功能改为默认，即所有计算 p=2 时都通过矩阵运算加速。

## 底层OP设计
无，通过已有的算子在 python 端进行组合。
## API实现方案
通过已有的算子在 python 端进行组合。通过对 x 和 y 的参数进行拼接扩展，之后将扩展后的 x 和 y 根据公式计算距离。
实现位置为 Paddle repo `python/paddle/tensor/linalg.py` 目录。
# 六、测试和验收的考量
测试 api 功能的准确性。
# 七、可行性分析和排期规划
本 API 难度适中，工期上能满足要求。
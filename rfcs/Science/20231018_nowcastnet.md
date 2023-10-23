# NowcastNet 设计文档

|              |                            |
|--------------|----------------------------|
| 提交作者     | co63oc                     |
| 提交时间     | 2023-10-18                 |
| RFC 版本号   | v1.0                       |
| 依赖飞桨版本 | develop/release 2.5.0 版本 |
| 文件名       | 20231018_nowcastnet.md       |

## 1. 概述

### 1.1 相关背景

[No.61：Skillful nowcasting of extreme precipitation with NowcastNet](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no61skillful-nowcasting-of-extreme-precipitation-with-nowcastnet)

极端降水是气象灾害的重要致灾因素，因此，通过具有高分辨率、长预警时间和局部细节的精准短临预报来减轻其对社会经济的影响，具有非常重要的意义。目前的方法存在模糊、散射、强度或位置误差等问题，基于物理的数值方法难以捕捉关键的混沌动力学，如对流起始，而基于数据的学习方法不能遵循固有的物理规律如平流守恒。
近年来，深度学习方法已被应用于天气预报，尤其是雷达观测的降水预报。这些方法利用大量雷达复合观测数据来训练神经网络模型，以端到端的方式进行训练，无需明确参考降水过程的物理定律。
这里复现了一个针对极端降水的非线性短临预报模型——NowcastNet，该模型将物理演变方案和条件学习法统一到一个神经网络框架中，实现了端到端预报误差优化。

### 1.2 功能目标

原仓库代码没有训练和评估过程，使用的为预训练权重，结果为生成预测图片，修改为转换预训练权重为 paddle 格式，使用 paddle 运行生成图片。
复现 mrms_case 和 mrms_large_case，并合入 PaddleScience

### 1.3 意义

复现 NowcastNet 模型，能够使用 NowcastNet 模型进行推理。

## 2. PaddleScience 现状

PaddleScience 套件暂无 NowcastNet 模型案例。

## 3. 目标调研

参考代码 https://codeocean.com/capsule/3935105/tree/v1
论文链接 https://www.nature.com/articles/s41586-023-06184-4#Abs1

原代码为 Pytorch 代码，需要在 PaddleScience 中复现，复现的主要问题是模型的转换，数据集读取的转换，使用 PaddleScience 的 API 调用。

## 4. 设计思路与实现方案

参考已有代码实现 NowcastNet
1. 模型构建
2. 读取预训练权重
3. 超参数设定
4. 生成预测图片

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

原代码使用脚本 mrms_case_test.sh，mrms_large_case_test.sh 运行，复现需要运行同样脚本生成相应图片，并使用 PaddleScience 复现

## 6. 可行性分析和排期规划

参考代码修改为 paddle 实现，使用 PaddleScience API，测试精度对齐
202309：调研
202310：基于 Paddle API 的复现，基于 PaddleScience 的复现
202311：整理项目产出，撰写案例文档

## 7. 影响面

在 ppsci.arch 下新增 NowcastNet 模型

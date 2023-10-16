# Null-text Inversion 设计文档

|              |                            |
|--------------|----------------------------|
| 提交作者     | co63oc                     |
| 提交时间     | 2023-10-17                 |
| RFC 版本号   | v1.0                       |
| 依赖飞桨版本 | develop/release 2.5.0 版本 |
| 文件名       | 20231017_nulltext_inversion.md       |

## 1. 概述

### 1.1 相关背景

[赛题五：复现图像编辑论文Null-text Inversion](https://competition.atomgit.com/competitionInfo?id=85216ad0ef0811ed99d49fc42bfa011c)

使用文本引导的扩散模型进行图像合成的进展因其卓越的真实感和多样性而备受关注。大规模模型激发了无数用户的想象力，这引发了持续的研究努力，探讨如何利用这些强大的模型进行图像编辑。最近，对合成的图像进行了直观的文本编辑，使用户能够仅通过文本轻松地操作图像。

### 1.2 功能目标

复现图像编辑论文 Null-text Inversion

DDIM 反演包括以相反的顺序执行 DDIM 采样。尽管每一步都会引入轻微的误差，但在无条件情况下效果很好。然而，在实践中，对于文本引导的合成来说，由于无分类器指导放大了其累积误差，因此这种方法会失效。我们观察到它仍然可以提供一个有希望的起始点进行反演。受GAN文献的启发，我们使用从初始 DDIM 反演获得的有噪声潜在代码序列作为关键点。然后，我们围绕这个关键点进行优化，以得到更好、更准确的反演。我们把这个高效优化过程称为 Diffusion Pivotal Inversion，它与现有工作的目标是不同的，后者旨在将所有可能的噪声向量映射到单个图像上。

### 1.3 意义

复现 Null-text Inversion 模型，能够使用 Null-text Inversion 模型进行推理。

## 2. PaddleMIX 现状

PaddleMIX 套件暂无 Null-text Inversion 模型案例。

## 3. 目标调研

参考代码 https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images
论文链接 https://arxiv.org/pdf/2211.09794.pdf

原代码为 Pytorch 代码，需要在 PaddleMIX 中复现，复现的主要问题是模型的转换，数据集读取的转换，复现相应结果。

## 4. 设计思路与实现方案

参考已有代码实现 Null-text Inversion
1. 模型构建
2. 数据构建
3. 计算域构建
4. 约束构建
5. 超参数设定
6. 优化器构建
7. 评估器构建
8. 模型训练、评估

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

原代码使用 null_text_w_ptp.ipynb 运行，复现需要达到原有模型结构，并生成相应结果图片。

## 6. 可行性分析和排期规划

202309：调研
202310：基于 Paddle API 的复现
202311：整理项目产出，撰写案例文档

## 7. 影响面

在 PaddleMIX/ppdiffusers/examples 下新增 Null-text Inversion 案例

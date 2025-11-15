# Discovering Symbolic Models from Deep Learning with Inductive Biases 设计文档 

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |xiaoyewww             |
| 提交时间      |2025-09-28          |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本   | develop            |
| 文件名        |hackthon9th_106_symbolic_deep_learning.md   |

## 1. 概述

### 1.1 相关背景

> 论文提出了一种通用的框架，将深度学习的效率与符号回归的解释性相结合。该方法利用具有强归纳偏差的深度学习模型（如图神经网络/Graph Neural Networks, GNNs）来学习数据的表示 。GNNs 因其适用于粒子系统等问题而被选择，它们天生具有物理学中常见的归纳偏差，例如粒子排列下的等变性。通过在 GNN 的训练过程中，强制其潜在表示（latent representations）稀疏或低维（例如使用L1正则化），可以使得模型学习到更有意义的、可解释的内部特征（例如将消息分量学习为力的线性组合）。然后，对学习模型内部的各个组件应用传统的符号回归，以提取明确的代数函数 。

### 1.2 功能目标

> 在本任务中，作者根据[Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)中的[代码](https://github.com/MilesCranmer/symbolic_deep_learning/tree/master)，复现symbolic_deep_learning训练。

### 1.3 意义

> 该论文为可解释的机器学习（eXplainable AI, XAI）和科学发现提供了一个强大的框架，它弥合了高效的“黑箱”深度学习和可解释的解析模型之间的鸿沟。

> 1. 可解释性和原理发现（Interpretability and Discovery）
> > 该方法提供了一种解释神经网络和从其学习的表示中发现新物理原理的替代方向。
> > 它成功地从神经网络中提取了已知正确的方程，包括力定律（Force Laws）和哈密顿量（Hamiltonians），实现了对既有物理规律的重新发现。
> > 在非平凡的宇宙学问题——暗物质模拟中，该方法发现了一个新的解析公式，可以根据附近宇宙结构的质量分布来预测暗物质晕的浓度这证明了该方法对未知物理规律的发现能力。

> 2. 泛化能力和扩展性（Generalization and Scalability）
> > 通过将深度学习模型分解为可解释的子组件，论文有效地将传统符号回归在高维数据集上的不可处理问题分解成了更小、可处理的子问题，从而扩展了符号回归的应用范围。
> > 研究发现，从 GNN 中提取的符号表达式，在处理分布外数据（out-of-distribution data）时，其泛化能力优于提取它的 GNN 本身。这暗示了简单符号模型在描述宇宙方面具有惊人的有效性。

## 2. PaddleScience 现状

> PaddleScience目前没有该模型的实现。

## 3. 目标调研

复现symbolic_deep_learning模型训练，精度与论文中对齐。

## 4. 设计思路与实现方案

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写。 

## 5. 测试和验收的考量

1. 完成Paddle后端和Pytorch/JAX后端模型精度对齐，按照PaddleCFD智能流体开发套件的单文件夹策略组织模型代码并合入PaddleCFD代码仓库。
2. 提供详细的模型精度对齐文档、数据集、模型训练说明文档。

## 6. 可行性分析和排期规划

1. 提交RFC 25年9月
2. 完成PR合入 25年10月

## 7. 影响面

无。
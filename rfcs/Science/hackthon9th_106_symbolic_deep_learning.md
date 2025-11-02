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

> 论文提出了一种通用框架，用于从具备强归纳偏置（inductive biases）的深度学习模型中提取符号化表达式。作者聚焦于图神经网络（Graph Neural Networks, GNNs），因其天然适用于粒子系统等具有关系结构的问题，并内置了物理系统中常见的对称性（如粒子置换不变性、空间等变性）。

框架包含两个阶段：

1. 稀疏表示学习阶段：在监督训练 GNN 时，通过 L1 正则化或 KL 散度约束，强制其消息传递模块（message function φₑ）输出低维且稀疏的潜在表示。实验表明，在这种约束下，GNN 学习到的消息向量会自动对齐真实物理力（或能量）所在的线性子空间，即消息 ≈ 线性变换 × 真实力。

2. 符号回归提取阶段：从训练好的 GNN 中提取 φₑ 等内部模块的输入-输出对（如粒子属性、相对位置 → 消息分量），并使用符号回归工具（如 PySR 或 Eureqa）拟合简洁的代数表达式。最终，这些符号表达式可替换原神经子模块，构成完全解析的可解释模型。

### 1.2 功能目标

> 在本任务中，作者根据[Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)中的[代码](https://github.com/MilesCranmer/symbolic_deep_learning/tree/master)，复现symbolic_deep_learning训练。

### 1.3 意义

> 本工作为可解释人工智能（XAI）与科学发现提供了一个兼具理论深度与实用价值的新范式，其核心意义体现在以下三方面：
>
> 1. **可解释性与物理规律发现**（Interpretability & Scientific Discovery）  
>    - 首次实证表明：GNN 的内部消息在稀疏约束下可**无监督地对齐真实物理力**（R² ≥ 0.97），为神经网络的“力学习”提供了直接证据。  
>    - 成功**重新发现**了多种形式的已知力定律（如 $F \propto 1/r^2$、弹簧力、带电粒子库仑力）与哈密顿量（如 $H_{\text{pair}} \propto q_1 q_2 / r$）。  
>    - 在暗物质模拟中**发现新公式**：$\hat{\delta}_i = C_1 + \frac{\sum_j (C_4 + M_j)}{C_2 + (C_6 \|r_i - r_j\|)^{C_7}}$，其预测误差（MAE=0.0882）显著优于领域专家设计的启发式公式（MAE=0.121）。
>
> 2. **高维符号回归的可扩展性**（Scalability of Symbolic Regression）  
>    - 通过将高维端到端模型**分解为低维可解释子模块**（φₑ, φᵥ, φᵤ），将传统符号回归难以处理的高维问题（如 100 维时间序列）转化为多个低维子问题，**指数级降低搜索空间**。
>    - 证明该框架可与**任意低维符号回归工具**（如 PySR、AI Feynman、gplearn）结合，具备良好通用性。
>
> 3. **优异的分布外泛化能力**（Out-of-Distribution Generalization）  
>    - 在暗物质任务中，对高过密度（δ > 1）的 OOD 数据：  
>      - GNN 本身测试 MAE = **0.142**  
>      - 提取出的符号模型测试 MAE = **0.0892**  
>    - 表明**简洁符号模型在描述物理世界时具有更强泛化性**，呼应了 Eugene Wigner 所述“数学在自然科学中不可思议的有效性”。

## 2. PaddleScience 现状

> PaddleScience目前没有该模型的实现。

## 3. 目标调研

复现符号回归模型训练，精度与论文中对齐。

## 4. 设计思路与实现方案

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写。 

## 5. 测试和验收的考量

1. 通过 PaddleScience 的代码风格检查
2. 符号回归训练精度与论文中对齐

## 6. 可行性分析和排期规划

1. 提交RFC 25年11月
2. 完成PR合入 25年11月

## 7. 影响面

无。
# Science 53 设计文档 

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |xiaoyewww             |
| 提交时间      |2025-03-17          |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        |hackthon8th_16_data_efficient_nopt.md   |

## 1. 概述

### 1.1 相关背景

> data_efficient_nopt旨在提高偏微分方程（PDE）算子学习的数据效率，通过设计无监督预训练方法减少对高成本模拟数据的依赖。利用未标记的PDE数据（无需模拟解），并通过基于物理启发的重建代理任务对神经算子进行预训练。为了提升分布外（OOD）泛化性能，我们进一步引入了一种基于相似性的上下文学习方法，使神经算子能够灵活利用上下文示例，而无需额外的训练成本或设计。在多种PDE上的实验表明，该方法具有高度的数据效率、更强的泛化能力，甚至优于传统的视觉预训练模型。

### 1.2 功能目标

> 在本任务中，作者根据[Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning
](https://arxiv.org/abs/2402.15734)中的[代码](https://github.com/delta-lab-ai/data_efficient_nopt)
>
> 复现data_efficient_nopt训练推理。

### 1.3 意义

> 无监督预训练：利用未标记的PDE数据进行无监督预训练，显著减少对模拟数据的依赖，性能优于基于模拟数据训练的模型或其他基准预训练模型。

> 可扩展的上下文学习：提出基于相似性的方法，提升神经算子的OOD泛化能力，可灵活扩展到大量未见过的上下文示例。

> 全面实验验证：在多样化的PDE基准和实际场景中进行了详细评估，展示了在正向建模性能和节省PDE模拟成本方面的显著优势。

## 2. PaddleScience 现状

> PaddleScience目前没有该模型的实现。

## 3. 目标调研

复现data_efficient_nopt模型训练和推理，精度与论文中对齐。

## 4. 设计思路与实现方案

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写。 

## 5. 测试和验收的考量

1. 通过 PaddleScience 的代码风格检查
2. data_efficient_nopt训练推理精度与论文中对齐

## 6. 可行性分析和排期规划

1. 提交RFC 25年3月
2. 完成PR合入 25年4月

## 7. 影响面

无。
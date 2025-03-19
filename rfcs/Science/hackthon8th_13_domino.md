# Science 53 设计文档 

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |xiaoyewww             |
| 提交时间      |2025-03-06          |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        |hackthon8th_13_domino.md   |

## 1. 概述

### 1.1 相关背景

> 外部空气动力学涉及高雷诺数Navier-Stokes方程求解，传统CFD方法计算成本高昂。神经算子通过端到端映射提升了效率，但面临多尺度耦合建模与长期预测稳定性不足的挑战。Decomposable Multi-scale Iterative Neural Operator（Domino）提出可分解多尺度架构，通过分层特征解耦、迭代残差校正及参数独立编码，显著提升跨尺度流动建模精度与泛化能力。实验显示，其计算速度较CFD快2-3个量级，分离流预测精度较FNO等模型提升约40%，为飞行器设计等工程问题提供高效解决方案。

### 1.2 功能目标

> 在本任务中，作者根据[DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)中的代码[Domino code](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/domino)
>
> 复现domino训练推理。

### 1.3 意义

> Domino计算速度较CFD快2-3个量级，分离流预测精度较FNO等模型提升约40%，为飞行器设计等工程问题提供高效解决方案。

## 2. PaddleScience 现状

> PaddleScience目前没有该模型的实现。

## 3. 目标调研

复现Domino模型训练推理，精度与论文中对齐。

## 4. 设计思路与实现方案

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写。 

### 4.1 技术难点

1. 开源项目中未提供预训练权重
2. 论文数据集较大，且前后处理所需要的内存较大

## 5. 测试和验收的考量

1. 通过 PaddleScience 的代码风格检查
2. Domino训练推理精度与论文中对齐

## 6. 可行性分析和排期规划

1. 提交RFC 25年3月
2. 完成PR合入 25年3月

## 7. 影响面

无。
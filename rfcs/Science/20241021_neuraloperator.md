# Nueraloperator 设计文档

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |       xiaoyewww    |
| 提交时间      |       2024-10-21   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | release 2.6.0 版本    |
| 文件名       | 20241021_neuraloperator.md |

## 1. 概述

### 1.1 相关背景

[【飞桨科学计算工具组件开发大赛】](https://github.com/PaddlePaddle/PaddleScience/issues/1000)适配Neural Operator。

[`neuraloperator`](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)是一种特殊的神经网络架构，称为神经算子，它不同于传统的神经网络仅能在有限维空间中进行映射，而是能够在无限维的函数空间中进行学习。这使得神经算子能够处理复杂的物理现象和动态系统，如流体力学、天气预测等，这些系统通常需要求解大规模的偏微分方程。

`neuraloperator`根据论文[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)和[Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481)，实现了一种通用神经网络去学习神经算子，以求解无限维函数空间的映射关系。

### 1.2 功能目标

> 具体需要实现的功能点。

1. 使用 paddle 的 python API 等价组合实现`neuraloperator`公开API的功能
2. 撰写 paddle 后端的单测文件，并自测通过

### 1.3 意义

> 简述本 RFC 所实现的内容的实际价值/意义。

不同于传统的神经网络仅能在有限维空间中进行映射，而是能够在无限维的函数空间中进行学习。这使得神经算子能够处理复杂的物理现象和动态系统，如流体力学、天气预测等，这些系统通常需要求解大规模的偏微分方程。

## 2. PaddleScience 现状

> 说明 PaddleScience 套件与本设计方案相关的现状。

当前的PaddleScience有`neuraloperator`相关的神经算子实现，参考[[Hackathon 6th Code Camp No.15] support neuraloperator](https://github.com/PaddlePaddle/PaddleScience/pull/867)和[PINO论文复现](https://github.com/PaddlePaddle/PaddleScience/pull/630), 按照现有代码补充或者代码重构。

## 3. 目标调研

> 如果是论文复现任务，则需说明复现论文解决的问题、所提出的方法以、复现目标以及可能存在的难点；如果是 API 开发类任务，则需说明已有开源代码中类似功能的实现方法，并总结优缺点。

1. 参考代码: https://github.com/neuraloperator/neuraloperator/tree/0.3.0

2. 参考论文：
    - [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
    - [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481)

原代码为 Pytorch 代码，需要在 PaddleScience 中复现，复现的主要问题是实现神经算子，使用 PaddleScience 的 API 调用，撰写飞桨后端的单测文件，并测试通过。

## 4. 设计思路与实现方案

> 结合 PaddleScience 套件的代码结构现状，描述如何逐步完成论文复现/API开发任务，并给出必要的代码辅助说明。

参考已有代码实现`neuraloperator`:

1. 模型构建，参考代码中主要实现的神经算子包括:

    - fno
    - fnogno
    - uno

2. 撰写单测文件，并测试通过。

### 4.1 补充说明[可选]

> 可以是对设计方案中的名词、概念等内容的详细解释。

参照赛事要求，所有文件组织结构必须与原有代码保持一致（新增文件除外），原有的注释、换行、空格、开源协议等内容不能随意变动（新增内容除外），否则会严重影响代码合入和比赛结束后成果代码的维护，因此不改动已有的相关代码，将代码实现在`ppsci.contrib.neuralop`中。

原有代码的设计如下:

```
neuraloperator
...
├── examples
│   ├── README.rst
│   ├── a.png
│   ├── checkpoint_FNO_darcy.py
│   ├── plot_FNO_darcy.py
│   ├── plot_SFNO_swe.py
│   ├── plot_UNO_darcy.py
│   ├── plot_darcy_flow.py
│   └── plot_darcy_flow_spectrum.py
...
├── neuralop
│   ├── __init__.py
│   ├── __pycache__
│   ├── datasets
│   ├── layers
│   ├── losses
│   ├── models
│   ├── mpu
│   ├── tests
│   ├── training
│   └── utils.py
...
```

设计后PaddleScience的`nerualop`代码结构如下:

```
PaddleScience
├── examples
│   ├── neuralop
...
├── ppsci
│   ├── contrib
│   │   ├── neuralop
│   │   │   ├── ...
...

```

## 5. 测试和验收的考量

> 说明如何对复现代码/开发API进行测试，以及验收标准，保障任务完成质量。

依据论文和参考代码实现神经算子，在examples文件夹中补充相应的例子和配置。

## 6. 可行性分析和排期规划

> 可以以里程碑列表的方式，细化开发过程，并给出排期规划。

参考代码修改为 paddle 实现，使用 PaddleScience API，测试精度对齐

- 2024.10.20~2024.10.21 提交RFC
- 2024.10.21~2024.11.11 完成案例代码的编写和调试，撰写单测文件。

## 7. 影响面

> 描述本方案对 PaddleScience 可能产生的影响。

- 影响到`ppsci/arch`，在其中新增多个`neuraloperator`模型。
- 影响到`examples`，在其中新增与原文接近的试验配置。

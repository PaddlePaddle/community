# GraphCast 设计文档

|              |                            |
|--------------|----------------------------|
| 提交作者     | DrownFish19                     |
| 提交时间     | 2023-12-01                 |
| RFC 版本号   | v1.0                       |
| 依赖飞桨版本 | develop/release 2.5.2 版本 |
| 文件名       | 20231201_graphcast.md       |

## 1. 概述

### 1.1 相关背景

[No.62：GraphCast: Learning skillful medium-range global weather forecasting](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/【PaddlePaddle%20Hackathon%205th】开源贡献个人挑战赛科学计算任务合集.md#no62graphcast-learning-skillful-medium-range-global-weather-forecasting)

GraphCast这种方法是天气预报领域的一项重大进展，它利用机器学习的能力来提高预测的准确性和效率。GraphCast通过图神经网络（GNNs）建模复杂的天气动态，并在欧洲中期天气预报中心（ECMWF）的ERA5再分析数据集上进行训练。它在全球范围内以0.25°的高分辨率快速预测数百种天气变量，并在多项目标上超越了ECMWF的高分辨率预测系统（HRES）。这项研究表明，GraphCast不仅能提高标准天气预测的效率，还在预测严重天气事件方面显示出潜力，可能对依赖天气的决策过程产生重大影响。

### 1.2 功能目标

* (任务要求目标) 复现graphcast模型，能够使用参考代码中提供的预训练权重进行推理。
* (后续计划目标）实现训练过程，并使用PaddleScience提供的测试数据集进行评估。

### 1.3 意义

复现 GraphCast 模型，能够使用 GraphCast 模型进行推理。

## 2. PaddleScience 现状

PaddleScience 套件暂无 GraphCast 模型案例。

## 3. 目标调研

参考代码 https://github.com/deepmind/graphcast
论文链接 https://arxiv.org/abs/2212.12794

原代码为 jax 代码，需要在 PaddleScience 中复现，复现的主要问题是模型转换、数据转换、jax特定函数转换。此处给出参考代码中各文件的功能说明。

*   `autoregressive.py`: Wrapper used to run (and train) the one-step GraphCast
    to produce a sequence of predictions by auto-regressively feeding the
    outputs back as inputs at each step, in JAX a differentiable way.

*   `casting.py`: Wrapper used around GraphCast to make it work using
    BFloat16 precision.

*   `checkpoint.py`: Utils to serialize and deserialize trees.
*   `data_utils.py`: Utils for data preprocessing.
*   `deep_typed_graph_net.py`: General purpose deep graph neural network (GNN)
    that operates on `TypedGraph` 's where both inputs and outputs are flat
    vectors of features for each of the nodes and edges. `graphcast.py` uses
    three of these for the Grid2Mesh GNN, the Multi-mesh GNN and the Mesh2Grid
    GNN, respectively.

*   `graphcast.py`: The main GraphCast model architecture for one-step of
    predictions.

*   `grid_mesh_connectivity.py`: Tools for converting between regular grids on a
    sphere and triangular meshes.

*   `icosahedral_mesh.py`: Definition of an icosahedral multi-mesh.
*   `losses.py`: Loss computations, including latitude-weighting.
*   `model_utils.py`: Utilities to produce flat node and edge vector features
    from input grid data, and to manipulate the node output vectors back
    into a multilevel grid data.

*   `normalization.py`: Wrapper for the one-step GraphCast used to normalize
    inputs according to historical values, and targets according to historical
    time differences.

*   `predictor_base.py`: Defines the interface of the predictor, which GraphCast
    and all of the wrappers implement.

*   `rollout.py`: Similar to `autoregressive.py` but used only at inference time
    using a python loop to produce longer, but non-differentiable trajectories.

*   `typed_graph.py`: Definition of `TypedGraph`'s.
*   `typed_graph_net.py`: Implementation of simple graph neural network
    building blocks defined over `TypedGraph` 's that can be combined to build
    deeper models.

*   `xarray_jax.py`: A wrapper to let JAX work with `xarray`s.
*   `xarray_tree.py`: An implementation of tree.map_structure that works with
`xarray` s.

## 4. 设计思路与实现方案

参考已有代码实现 GraphCast
1. 模型转换
2. 数据读取方式转换
3. loss计算方式转换
4. 模型推理，当前计划不设计模型训练过程，但提供可训练代码

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

依据论文给定权重，加载模型权重后相同输出下对齐。

## 6. 可行性分析和排期规划

参考代码修改为 paddle 实现，使用 PaddleScience API，测试精度对齐
* 20231201：调研
* 20231202-20231210：基于 Paddle API 的复现，基于 PaddleScience 的复现
* 20231210-20231215：整理项目产出，撰写案例文档

## 7. 影响面

* 第一阶段考虑在PaddleScience/jointContribution中添加
* 第二阶段将模型迁移至PaddleScience模型库中，并给出训练相关完善代码。

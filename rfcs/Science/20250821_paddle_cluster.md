# paddle-cluster 设计文档

> RFC 文档相关记录信息

|              |                    |
| ------------ | ------------------ |
| 提交作者     | BeingGod          |
| 提交时间     | 2025-08-21         |
| RFC 版本号   | v1.0               |
| 依赖飞桨版本 | 3.1 版本 |
| 文件名       | 20250821_paddle_cluster |

## 1. 概述

### 1.1 相关背景

[第三方生态库pytorch_cluster适配](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91FundableProject%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E4%B8%80%E7%AC%AC%E4%B8%89%E6%96%B9%E7%94%9F%E6%80%81%E5%BA%93pytorch_cluster%E9%80%82%E9%85%8D)

### 1.2 功能目标

1. 完整实现pytorch_cluster中的全部API，精度实现对齐；
2. 实现对应的单元测试；


### 1.3 意义

pytorch_cluster包含用于PyTorch的高度优化的图聚类算法，是知名图神经网络库pytorch_geometric底层基础库之一，为了完整实现pytorch_geometric需要实现实现pytorch_cluster。

## 2. 目标调研

### 2.1 适配版本

|      组件     |  版本    |
| ------------ | ------------------ |
| paddle | 3.1 |
| pytorch-cluster | 3d1d9e3 |


### 2.2 相关接口及对应单测梳理


|  C++ 算子 | python 层接口     |  对应单测 |
| ------------ | ------------------ |------------------ |
| fps | fps | test_fps.py |
| graclus | graclus_cluster | test_graclus.py |
| grid | grid_cluster | test_grid.py |
| knn | knn, knn_graph | test_knn.py|
| nearest | nearest | test_nearest.py|
| radius | radius, radius_graph| test_radius.py|
| random_walk | random_walk | test_rw.py |
| neighbor_sampler | neighbor_sampler | test_sampler.py|

## 3. 设计思路与实现方案

1. 梳理依赖算子及对应单测
2. 完成 demo 算子接入，跑通编译
3. 完成 C++ 算子接入
4. 完成 python 接口改写，跑通对应单测
5. 基于实际 workload 对 paddle 进行 benchmark，优化性能瓶颈，性能对齐 torch

### 3.1 补充说明[可选]

由于 Paddle 主框架大部分 CPU kernel 不支持 bf16 和 fp16 精度，paddle-cluster 中跳过 fp16和bf16 精度的 CPU 单测。

## 4. 测试和验收的考量

1. 实现pytorch_cluster中的API，并给出安装、使用说明；
2. 实现 其中对应的单元测试并通过；
3. 最终代码合入PFCCLab组织下；

## 5. 可行性分析和排期规划

- 2025.8：调研，复现代码并作调整
- 2025.9：整理项目产出，撰写案例文档

## 6. 影响面

1. 在 PFCC 下添加 paddle_cluster 仓库

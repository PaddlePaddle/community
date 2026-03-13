# SchNet 设计文档

> RFC 文档相关记录信息

|              |                       |
| ------------ | --------------------- |
| 提交作者     | decade-afk             |
| 提交时间     | 2026-02-27            |
| RFC 版本号   | v1.0                  |
| 依赖飞桨版本 | release 3.3 版本      |
| 文件名       | `hackathon10th_18_schnet.md` |

## 1. 概述

### 1.1 相关背景

[NO.18 SchNet 论文复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%98%A5%E8%8A%82%E7%89%B9%E5%88%AB%E5%AD%A3%E2%80%94%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no18--SchNet%E6%A8%A1%E5%9E%8B%E5%A4%8D%E7%8E%B0)

SchNet 是经典原子体系图网络模型，主要用于分子/材料的能量与力预测。当前需要将[论文](https://www.nature.com/articles/ncomms13890)中所提及的 `SchNet` 模型迁移到 PaddleMaterials，并纳入统一训练与推理框架。

### 1.2 功能目标

1. 在 PaddleMaterials 中完成 SchNet 模型复现，支持统一 `trainer/predictor/sampler`。
2. 覆盖论文与原始实现常用数据集：QM9、MD17、ISO17。
3. 完成前向、反向、数据处理、训练配置、指标的全链路对齐。
4. 输出可复现实验文档、配置、日志与模型文件。

### 1.3 意义

补齐 PaddleMaterials 在小分子方向（interatomic potentials）中的 SchNet 案例，提升分子性质预测与势能面建模能力，形成可复用基线。

## 2. PaddleMaterials 现状

1. 主干框架已具备统一训练器、数据集工厂、图构建工厂能力。
2. 需要在 `ppmat/models`、`ppmat/datasets`、`interatomic_potentials/configs` 中补齐并标准化 SchNet 实现与案例。
3. 需保证新任务文档、数据链接、评估脚本齐全。

## 3. 目标调研

1. 论文解决的问题：通过连续滤波卷积建模原子相互作用，实现分子能量/力等性质预测。
2. 参考代码：
   `https://github.com/atomistic-machine-learning/SchNet`
   `https://github.com/atomistic-machine-learning/schnetpack`
3. 参考论文：
   `https://arxiv.org/abs/1706.08566`
4. 核心迁移要求：PyTorch/TensorFlow 版本到 Paddle 版本的结构等价、训练等价、指标等价。

## 4. 设计思路与实现方案

1. 模型实现：模型文件放置在 `ppmat/models/schnet`，对齐 SchNet 结构参数（`n_atom_basis`、`n_interactions`、`n_rbf`、`cutoff`、`readout`）。
2. 数据处理：数据处理使用已有工厂函数与统一 dataset 接口；QM9/MD17/ISO17 使用统一 split 管理与缓存机制。
3. 数值对齐：先做单 batch 前向 logits 对齐，再做梯度对齐，再做短程训练 loss 轨迹对齐，最后做全量指标对齐。
4. 训练配置对齐：优先对齐原始仓库配置口径，包括学习率策略、batch size、目标量单位、offset/atomref 处理方式。
5. 框架接入：训练走 `interatomic_potentials/train.py`，预测走统一 predictor，采样/生成类任务接口保持与套件一致。

## 5. 测试和验收考量

### 5.1 验收标准

1. **单卡前向精度对齐**：前向 logits diff 1e-4 量级（生成式 1e-6）。
2. **反向对齐**：训练 2 轮以上，loss 一致。
3. **训练精度对齐**：ImageNet 数据集精度 diff 0.2% 以内（若任务适用）。
4. **监督类任务**：metric 误差控制在 1% 以内。
5. **生成式模型**：采样指标误差 5% 以内。

### 5.2 备注说明

1. 模型文件放到 `ppmat/models`，采用统一 `trainer/predictor/sampler`。
2. 数据集/预训练模型/log 原始文件通过百度网盘链路交付，并将最终百度云链接写入 dataset/model/readme。
3. 扩散模型相关组件放在 `ppmat/scheduler`。
4. 套件中暂无的任务类型需新增任务 readme 与任务类型说明文档。
5. 数据处理需使用已有 `build_structure/build_molecule` 等工厂函数。
6. 每个复现 PR 需覆盖原论文所有数据集，并给出训练与推理/采样指标对照。

## 6. 可行性分析与排期规划

1. 第 1 阶段：梳理论文、原始仓库、数据集与配置差异。
2. 第 2 阶段：完成模型、数据、配置、训练脚本接入。
3. 第 3 阶段：完成前向/反向/数据处理/指标四类对齐。
4. 第 4 阶段：完成完整训练、汇总报告、文档与案例说明。

## 7. 影响面

1. 新增/完善 SchNet 模型与配置，影响 `ppmat/models`、`ppmat/datasets`、`interatomic_potentials/configs`。
2. 扩展 PaddleMaterials 在小分子任务中的标准案例库与复现基线。

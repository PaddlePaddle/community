# NewtonNet 设计文档

> RFC 文档相关记录信息

|              |                                |
| ------------ | ------------------------------ |
| 提交作者     | co63oc                         |
| 提交时间     | 2026-04-08                     |
| RFC 版本号   | v1.0                           |
| 依赖飞桨版本 | develop 版本                   |
| 文件名       | `hackathon10th_9_newtonnet.md` |

## 1. 概述

### 1.1 相关背景

NewtonNet模型复现，模型迁移到 PaddleMaterials，并纳入统一训练与推理框架。

### 1.2 功能目标

1. 在 PaddleMaterials 中完成 NewtonNet 模型复现，支持统一 `trainer/predictor/sampler`。
2. 覆盖论文与原始实现数据集。
3. 完成前向、反向、数据处理、训练配置、指标的全链路对齐。
4. 输出可复现实验文档、配置、日志与模型文件。

### 1.3 意义

补齐 PaddleMaterials 在机器学习原子间势函数方向（interatomic potentials）中的 NewtonNet 案例，提升分子性质预测与势能面建模能力，形成可复用基线。

## 2. PaddleMaterials 现状

1. 主干框架已具备统一训练器、数据集工厂、图构建工厂能力。
2. 需要在 `ppmat/models`、`ppmat/datasets`、`interatomic_potentials/configs` 中补齐并标准化 NewtonNet 实现与案例。
3. 需保证新任务文档、数据链接、评估脚本齐全。

## 3. 目标调研

1. 参考代码：
   `https://github.com/THGLab/NewtonNet`
2. 核心迁移要求：PyTorch/TensorFlow 版本到 Paddle 版本的结构等价、训练等价、指标等价。

## 4. 设计思路与实现方案

1. 模型实现：模型文件放置在 `ppmat/models/newtonnet`，对齐 NewtonNet 结构参数。
2. 数据处理：数据处理使用已有工厂函数与统一 dataset 接口。
3. 数值对齐：先做单 batch 前向 logits 对齐，再做梯度对齐，再做短程训练 loss 轨迹对齐，最后做全量指标对齐。
4. 训练配置对齐：优先对齐原始仓库配置口径，包括学习率策略、batch size、目标量单位、offset 处理方式。
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
5. 数据处理需使用已有工厂函数。

## 6. 可行性分析与排期规划

1. 第 1 阶段：梳理论文、原始仓库、数据集与配置差异。
2. 第 2 阶段：完成模型、数据、配置、训练脚本接入。
3. 第 3 阶段：完成前向/反向/数据处理/指标四类对齐。
4. 第 4 阶段：完成完整训练、汇总报告、文档与案例说明。

## 7. 影响面

1. 新增/完善 Newtonnet 模型与配置，影响 `ppmat/models`、`ppmat/datasets`、`interatomic_potentials/configs`。
2. 扩展 PaddleMaterials 在机器学习原子间势函数任务中的标准案例库与复现基线。

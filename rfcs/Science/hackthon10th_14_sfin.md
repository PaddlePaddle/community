# SFIN 设计文档

> RFC 文档相关记录信息

|              |                    |
| ------------ | ------------------ |
| 提交作者     | ADream-ki          |
| 提交时间     | 2025-03-05         |
| RFC 版本号   | v1.0               |
| 依赖飞桨版本 | release 3.0 版本 |
| 文件名       | hackthon10th_14_sfin.md |

## 1. 概述

### 1.1 相关背景

[NO.14 SFIN 论文复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%98%A5%E8%8A%82%E7%89%B9%E5%88%AB%E5%AD%A3%E2%80%94%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no14--sfin%E6%A8%A1%E5%9E%8B%E5%A4%8D%E7%8E%B0)

SFIN (Scientific Foundation Model) 模型是面向科学计算的基础模型，旨在为科学计算领域提供强大的通用模型能力。该模型通过结合卷积型和科学数据处理技术，实现高效的科学数据生成和处理。

### 1.2 功能目标

1. 复现 SFIN 代码，实现完整的训练、推理、预测流程。
2. 保持精度与论文精度一致，相对误差在 ±1% 以内。
3. 产出论文相关文档、图片、视频等。

### 1.3 意义

复现 SFIN 代码，能够使用 SFIN 模型进行训练、推理、预测，丰富 PaddleMaterials 的科学计算模型库。

## 2. PaddleMaterials 现状

PaddleMaterials 套件暂无 SFIN 代码案例。

## 3. 目标调研

- 论文解决的问题：
  SFIN 模型解决科学计算中高效数据生成和处理的问题
- 链接：
  代码：[https://github.com/HeasonLee/SFIN](https://github.com/HeasonLee/SFIN)
  论文：[https://arxiv.org/pdf/2504.02555](https://arxiv.org/pdf/2504.02555)

  需将 Pytorch 代码转换为 Paddle

## 4. 设计思路与实现方案

参考 PaddleMaterials 已有代码实现 SFIN

1. 数据预处理
2. 模型构建
3. 超参数设定
4. 模型推理的评估指标

### 4.1 补充说明[可选]

目前SFIN模型未公开训练数据集以及训练配置，需联系作者提供

## 5. 测试和验收的考量

### 5.1 验收标准

- **单卡前向精度对齐**：前向logits diff 1e-4 量级（生成式1e-6）。
- **反向对齐**：训练2轮以上，loss一致。
- **训练精度对齐**：ImageNet数据集精度 diff 0.2%以内。
- **监督类任务**：metirc误差控制在1%以内
- **生成式模型**：采样指标保持误差5%以内

### 5.2 备注说明

- 模型文件放到 `ppmat/model` 下，所有模型需采用统一的 trainer/predictor/sampler
- 数据集/预训练模型文件/log原始文件通过百度网盘链接通过 pr/微信等方式给到百度工程师，工程师会给到百度云链接，需把相应的链接放到相应的dataset/model/readme等文件中
- 扩散模型相关组件在 `ppmat/scheduler` 里
- 对于套件里暂时还没有的任务，新模型对应的新任务，需新建相应的任务readme文件
- 数据处理模块，需使用已有的 `build_structure/build molecule` 等工厂函数
- 每个复现的模型PR需保证囊括了原论文的所有数据集，模型训练精度和推理/采样metric指标均可以对应原始论文
- 对应套件内还没有的任务类型，需要添加新增的任务类型说明readme文档

## 6. 可行性分析和排期规划

- 202503：调研
- 202504：复现代码并作调整
- 202504：整理项目产出，撰写案例文档

## 7. 影响面

丰富 PaddleMaterials 的应用案例，在 ppmat.model 中新增 SFIN model

此文档展示 **PaddlePaddle Hackathon 第九期活动——开源贡献个人挑战赛科学计算方向任务** 详细介绍

## 【开源贡献个人挑战赛-科学计算】任务详情

复现任务说明：

开发流程：

1. 要求基于 PaddleScience 套件进行开发，开发文档参考：[https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/](https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/)。
2. 复现整体流程和验收标准可以参考：[https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#21](https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#21)，复现完成后需供必要的训练产物，包括训练结束后保存的 train.log日志文件、.pdparams模型权重参数文件（可用网盘的方式提交）、撰写的.md案例文档。
3. 理解复现流程后，可以参考 PaddleScience 开发文档：[https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/](https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/)，了解各个模块如何进行开发、修改，以及参考API文档，了解各个现有API的功能和作用：[https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/](https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/)。
4. 案例文档撰写格式可参考 [https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d/](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d/)，最终合入后会被渲染并展示在 PaddleScience 官网文档。
5. 如在复现过程中出现需添加的功能无法兼容现有 PaddleScience API 体系（PaddleScience API 文档﻿），则可与论文复现指导人说明情况，并视情况允许直接基于 Paddle API 进行复现。
6. 若参考代码为 pytorch，则复现过程可以尝试使用 PaConvert（[https://github.com/PaddlePaddle/PaConvert](https://github.com/PaddlePaddle/PaConvert)）辅助完成代码转换工作，然后可以尝试使用 PaDiff（[https://github.com/PaddlePaddle/PaDiff](https://github.com/PaddlePaddle/PaDiff)）工具辅助完成前反向精度对齐，从而提高复现效率。

验收标准：参考模型复现指南验收标准部分[https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#3](https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#3)

### NO.103 基于Paddle实现第三方库e3nn

**详细描述：**

- 实现E(3)等变神经网络的第三方工具库e3nn
- 参考代码链接：[https://github.com/e3nn/e3nn](https://github.com/e3nn/e3nn)
- 相关实现：[https://github.com/PaddlePaddle/PaddleMaterials/tree/InfGCN/experimental/ppmat/models/common/e3nn](https://github.com/PaddlePaddle/PaddleMaterials/tree/InfGCN/experimental/ppmat/models/common/e3nn)

**验收标准**：

- 实现其中全部API。
- 完成单元测试
- 实现精度对齐。
- 安装文档。

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 Paddle 框架

### NO.104 基于Paddle实现第三方库torchmetrics

**详细描述：**

- 基于Paddle实现机器学习常用指标库torchmetrics

* 参考代码链接：[https://github.com/Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics)

**验收标准**：

- 实现其中全部API。
- 完成单元测试
- 实现精度对齐。
- 安装文档。

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 Paddle 框架

### NO.105 基于Paddle实现CoNFiLD流场生成模型

**详细描述：**

- 实现基于Paddle框架的CoNFiLD模型

* 论文链接：[https://www.nature.com/articles/s41467-024-54712-1](https://www.nature.com/articles/s41467-024-54712-1)
* 参考代码链接：[https://github.com/jx-wang-s-group/CoNFiLD](https://github.com/jx-wang-s-group/CoNFiL)

**验收标准**：

- 完成Paddle后端和Pytorch后端模型精度对齐，按照PaddleCFD智能流体开发套件的单文件夹策略组织模型代码并合入PaddleCFD代码仓库。
- 提供详细的模型精度对齐文档、数据集、模型训练说明文档。

### NO.106 基于Paddle实现符号深度学习模型，用于流体力学方程发现

**详细描述：**

- 基于Paddle实现符号深度学习模型，用于流体力学方程发现

* 论文链接：[https://proceedings.neurips.cc/paper/2020/file/c9f2f917078bd2db12f23c3b413d9cba-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/c9f2f917078bd2db12f23c3b413d9cba-Paper.pdf)
* 参考代码链接：[https://github.com/MilesCranmer/symbolic_deep_learning](https://github.com/MilesCranmer/symbolic_deep_learning)

**验收标准**：

- 完成Paddle后端和Pytorch/JAX后端模型精度对齐，按照PaddleCFD智能流体开发套件的单文件夹策略组织模型代码并合入PaddleCFD代码仓库。
- 提供详细的模型精度对齐文档、数据集、模型训练说明文档。

### NO.107 基于PaddleScience复现Aurora模型推理，使用小样本数据能够实现微调及训练

**论文链接：**

[https://doi.org/10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)
**代码复现：**

- 复现Aurora模型推理，使用小样本数据能够实现微调及训练，精度与源码对齐，并合入PaddleScience

**参考代码链接：**

[https://arxiv.org/pdf/2311.07222](https://arxiv.org/pdf/2311.07222)

### NO.108 基于PaddleScience复现neuralgcm模型推理，使用小样本数据能够实现训练

**论文链接：**

[https://doi.org/10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)

**代码复现：**

- 复现neuralgcm模型推理，使用小样本数据能够实现训练，精度与源码对齐，并合入PaddleScience

**参考代码链接：**

[https://github.com/google-research/neuralgcm](https://github.com/google-research/neuralgcm)

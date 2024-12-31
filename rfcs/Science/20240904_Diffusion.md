# NO.8 A physics-informed diffusion model for high-fidelity flow field reconstruction 论文复现

> RFC 文档相关记录信息

|              |                       |
| ------------ | --------------------- |
| 提交作者     | AI1LJW                |
| 提交时间     | 2024-9-10             |
| RFC 版本号   | v1.0                  |
| 依赖飞桨版本 | develop 版本          |
| 文件名       | 20240904_Diffusion.md |

## 1. 概述

### 1.1 相关背景

> 题目： [NO.8 A physics-informed diffusion model for high-fidelity flow field reconstruction 论文复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no8-a-physics-informed-diffusion-model-for-high-fidelity-flow-field-reconstruction-%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0)
>
> 计算流体动力学（CFD）模拟对工程系统与流体流动的相互作用提供有价值的信息，对工程设计和相关领域非常重要。然而，高保真度的CFD模拟，如直接数值模拟（DNS），通常需要在大规模的空间和时间尺度上求解纳维-斯托克斯方程，计算成本很高。而机器学习模型在加速CFD模拟方面显示出潜力，因此论文使用一种仅使用高保真度数据进行训练的扩散模型，用于重建高保真度数据，并且在实验结果表明，该模型能够在不重新训练的情况下，基于不同的输入源产生准确的2D湍流流动重建结果

### 1.2 功能目标

* 复现 DPMM 代码，实现完整的流程，包括：训练、验证、导出。
* 保持精度与论文精度一致，相对误差在 ±10% 以内。
* 产出论文相关文档、图片等。

### 1.3 意义

> 复现 DPMM 代码，能够使用 DPMM 模型进行训练、验证、导出。

## 2. PaddleScience 现状

> PaddleScience 套件暂无 DPMM 代码案例，但是可以基于PaddleScience API实现该模型。

## 3. 目标调研

> - 论文解决的问题：将数据重建问题转化为数据去噪问题，并使用去噪扩散概率模型 (DDPM) 从噪声输入中重建高精度 CFD 数据。
> - 链接：
>   代码：[https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution)
>   论文：[https://www.sciencedirect.com/science/article/pii/S0021999123000670](https://www.sciencedirect.com/science/article/pii/S0021999123000670)
>
>   需将 Pytorch 代码转换为 Paddle

## 4. 设计思路与实现方案

参考 PaddleScience 以及 AiStudio 已有代码实现 DPMM

1. 数据预处理
2. 模型构建
3. 超参数设定
4. 验证训练模型的 rel. error

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

> 成功复现DPMM模型，并在论文中的案例上复现精度。

## 6. 可行性分析和排期规划

- 202408：调研
- 202409：复现代码并作调整
- 202410：整理项目产出，撰写案例文档

## 7. 影响面

> 丰富[PaddleScience](https://paddlescience-docs.readthedocs.io/zh/latest/)的应用案例，在example目录下增加DPMM模型。

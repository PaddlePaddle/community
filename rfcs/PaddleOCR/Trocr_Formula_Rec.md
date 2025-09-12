# TrOCR-Formula-Rec——设计文档

| 任务名   | TrOCR-Formula-Rec    |
| -------- | -------------------- |
| 提交作者 | co63oc               |
| 提交时间 | 2025-05-13           |
| 版本号   | 1.0.0                |
| 依赖     | develop              |
| 文件名   | Trocr_Formula_Rec.md |

# 一、概述

## 1、相关背景

基于 TrOCR 算法、整合 UniMER-1M 等公式识别数据集的公式识别项目，该模型可以在本地离线环境中进行高效的 CPU 推理。

## 2、功能目标

本任务的目标是在 PaddleOCR 中复现 TrOCR-Formula-Rec 模型。复现性能指标与原始仓库效果相当，并在提交的文档中给出的 SPE-BLEU、SPE-EditDis、CPE-BLEU、CPE-EditDis、SCE-BLEU、SCE-EditDis、HWE-BLEU 和 HWE-EditDis 指标值。模型存储大小<300M，推理耗时与原始仓库相当，简单公式在1s左右。

## 3、意义

- 扩充PaddleOcr中的模型，用户可以使用更多 OCR 模型。

# 二、飞桨现状

[PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX/tree/develop) ppdiffusers 中没有 trocr, DeiT 和 VisionEncoderDecoder 模型结构，可以参考transformers库加入 trocr, DeiT 和 VisionEncoderDecoder 的模型结构。在PaddleOCR中调用 ppdiffusers 中的模型进行推理验证。

# 三、业内方案调研

1. pytorch框架环境下，trocr 在 transformers 中实现 [transformers/src/transformers/models/trocr](https://github.com/huggingface/transformers/tree/main/src/transformers/models/trocr)
2. pytorch框架环境下，DeiT 在 transformers 中实现 [transformers/src/transformers/models/deit](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deit)
3. pytorch框架环境下，VisionEncoderDecoder 在 transformers 中实现 [transformers/src/transformers/models/vision_encoder_decoder](https://github.com/huggingface/transformers/tree/main/src/transformers/models/vision_encoder_decoder)

# 四、对比分析

使用作者开源的模型和代码进行实现为目前最佳的论文复现实践方式，但基于控制影响面考量，将作者开源的模型转换成基于飞桨的模型，并为PaddleOCR增加研究中提出的数据生成和增强方式为目前的最佳实践方式

# 五、设计思路与实现方案

## 总体思路

1. 在 PaddleMIX 增加模型
2. 在 PaddleOcr 增加训练推理验证模块。

# 六、测试和验收的考量

参考论文中的测试方式，并在提交的文档中给出 SPE-BLEU、SPE-EditDis、CPE-BLEU、CPE-EditDis、SCE-BLEU、SCE-EditDis、HWE-BLEU 和 HWE-EditDis 具体值

# 七、可行性分析和排期规划

2025-05 实现模型复现，完成文档编写和提交。

# 八、影响面

- PaddleOCR 中增加trocr模型。

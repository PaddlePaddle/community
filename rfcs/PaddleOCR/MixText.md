# MixText——设计文档

| 任务名  | PaddleOcr--MixText |
| ---- | ------------------ |
| 提交作者 | ErnestinaQiu       |
| 提交时间 | 2024-10-22         |
| 版本号  | v1.0               |
| 依赖   | develop版本          |
| 文件名  | MixText.md         |

## 一、概述

## 1、相关背景

针对易混淆的文本公式，该研究主要提出了一种专门的多模态OCR模型和一种新颖的数据采集方法。多模态模型采用了Swin Transformer作为编码器提取视觉特征，采用了RoBERTa作为解码器通过融合语意结构提升识别率。值得一提的是，在识别清晰的图像时，模型更多地依赖于图像，而非上下文语意给出后续的识别结果。论文仓库中仅给出了数据生成的部分代码。

## 2、功能目标

1.在PaddleOcr中增加该模型与数据生成方法，复现性能指标与原始仓库效果相当，并在提交的文档中给出 Edit distance(Edit Dis.)、BLEU score、Precision、Recall 具体值。

## 3. 意义

1. 扩充PaddleOcr中的模型，给予用户更多选择

2. 扩充PaddleOcr中的数据生成方法

# 二、现状

PaddleOcr已对经典的ocr识别算法进行了支持，并持续优化通用识别模型。配套提供了数据生成和增强的工具。复现本研究的目的是为了进一步跟进前沿科技成果。

# 三、业内方案调研

1. 已开源数据生成代码至代码仓库  
   
   [RQLuo/MixTeX-Latex-OCR: MixTeX multimodal LaTeX, ZhEn, and, Table OCR. It performs efficient CPU-based inference in a local offline on Windows. (github.com)](https://github.com/RQLuo/MixTeX-Latex-OCR/tree/main)

2. 提供在线demo网址 
   
   [https://mineai.top/](https://mineai.top/)

3. Swin Transformer

[microsoft/swin-tiny-patch4-window7-224 · Hugging Face](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)

4. RoBERTa

    [fairseq/examples/roberta/README.md at main · facebookresearch/fairseq (github.com)](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md)

# 四、对比分析

相较于PaddleOcr中已有的LaTeX-OCR, MixText采用了nlp模型作为decoder，通过对特定上下文的学习和融合自然语义，为印刷体和手写体公式的识别率的提升提供了新的思路。

提出了新颖的数据集生成方法，并通过消融实验论证了，使用基于该方法的混合数据集的效果优于单纯使用合成数据集与真实数据集，为识别率指标提升提供了思路。

# 五、设计思路与实现方案

总体思路

1. 在PaddleOcr的PaddleOCR/ppocr/modeling模块增加MixTex模型，在PaddleOCR/ppocr/postprocess模块增加对应的后处理。

2. 在PaddleOCR/ppocr/data/中增加mixtex_dataset.py,在PaddleOCR/ppocr/data/imaug中增加数据生成方式

# 六、测试和验收的考量

参考论文中的测试方式，并在提交的文档中给出 Edit distance(Edit Dis.)、BLEU score、Precision、Recall 具体值

# 七、影响面

1. PaddleOCR/ppocr/data/中增加数据生成方法

2. 计划在PaddleOCR/ppocr中增加模型模块和其前后处理，并以增加参数可选值而非增加新参数的方式整合入现有的训练和推理pipeline.

# 八、排期规划

预计能在活动期内完成。
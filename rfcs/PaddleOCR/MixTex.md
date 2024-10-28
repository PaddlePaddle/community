# MixText——设计文档

| 任务名  | PaddleOcr--MixText |
| ---- | ------------------ |
| 提交作者 | ErnestinaQiu       |
| 提交时间 | 2024-10-26         |
| 版本号  | v1.0               |
| 依赖   | develop版本          |
| 文件名  | MixTex.md          |

# 一、概述

## 1、相关背景

针对易混淆的文本公式，该研究主要提出了一种多模态OCR模型和一种新颖的数据采集方法。多模态模型采用了Swin Transformer作为编码器提取视觉特征，采用了RoBERTa作为解码器通过融合语意结构提升识别率。值得一提的是，在识别清晰的图像时，模型更多地依赖于图像，而非上下文语意给出后续的识别结果。

## 2、功能目标

1.在PaddleOcr中增加该模型与数据生成方法，复现性能指标与原始仓库效果相当，并在提交的文档中给出 Edit distance(Edit Dis.)、BLEU score、Precision、Recall 具体值。

## 3、意义

- 扩充PaddleOcr中的模型，给予用户更多选择

- 扩充PaddleOcr中的数据生成方法

# 二、飞桨现状

PaddleOCR目前暂无对swin transformer encoder和RoBERTa的支持，可以通过PaddlePaddle.nn以现有api的组合方式实现。

# 三、业内方案调研

1. 已开源数据生成代码、模型训练和推理代码至代码仓库
   
   [RQLuo/MixTeX-Latex-OCR: MixTeX multimodal LaTeX, ZhEn, and, Table OCR. It performs efficient CPU-based inference in a local offline on Windows. (github.com)](https://github.com/RQLuo/MixTeX-Latex-OCR/tree/main)

2. Swin Transformer

[microsoft/swin-tiny-patch4-window7-224 · Hugging Face](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)

4. RoBERTa

    [fairseq/examples/roberta/README.md at main · facebookresearch/fairseq (github.com)

# 四、对比分析

使用作者开源的模型和代码进行实现为目前最佳的论文复现实践方式，但基于控制影响面考量，将作者开源的模型转换成基于飞浆的模型，并为PaddleOCR增加研究中提出的数据生成和增强方式为目前的最佳实践方式

# 五、设计思路与实现方案

## 总体思路

1. 在PaddleOCR/ppocr/data/中增加mixtex_dataset.py,在PaddleOCR/ppocr/data/imaug中增加数据生成方式

2. 在PaddleOcr的PaddleOCR/ppocr/modeling模块增加MixTex模型，在PaddleOCR/ppocr/postprocess模块增加对应的后处理。

# 六、测试和验收的考量

参考论文中的测试方式，并在提交的文档中给出 Edit distance(Edit Dis.)、BLEU score、Precision、Recall 具体值

# 七、可行性分析和排期规划

预计能在活动期内完成。

# 八、影响面

- PaddleOCR/ppocr/data/中增加数据生成方法

- 计划在PaddleOCR/ppocr中增加模型模块和其前后处理，并以增加参数可选值而非增加新参数的方式整合入现有的训练和推理pipeline.

# 名词解释

# 附件及参考资料
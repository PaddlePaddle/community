此文档展示 **PaddlePaddle Hackathon 第八期活动——开源贡献个人挑战赛套件开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-套件开发】任务详情

### NO.5 在 PaddleSpeech 中复现 DAC 模型

**详细描述：**

- 在 PaddleSpeech 套件中实现并对齐 Descript-Audio-Codec 的分布式训练、推理和评估流程。
- 相关论文：https://arxiv.org/abs/2306.06546
- 参考：https://github.com/descriptinc/descript-audio-codec

**验收标准**：

- 复现的性能指标需要与论文预期一致
- 需上传完整的训练代码和训练脚本以及模型
- 代码相关文档和注释完备

**技术要求：**

- 了解 DAC 模型
- 熟练掌握 Python 语言
- 熟悉 PaddleSpeech 框架及其数据处理流程

### NO.6 在 PaddleSpeech 中实现 Whisper 的 Finetune

**详细描述：**

- 相关论文：https://arxiv.org/pdf/2212.04356
- 参考：https://huggingface.co/blog/fine-tune-whisper
- 数据：https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
- 尽量复用已有推理代码：paddlespeech/s2t/models/whisper/whipser.py

**验收标准**：

- 复现的性能指标需要与原始仓库效果相当
- 需上传完整的训练代码和训练脚本
- 训练后的模型支持 command line 方式推理
- 支持原生 whisper large v3 通过 command line 方式推理
- 代码相关文档和注释完备

**技术要求：**

- 了解 Whisper 模型
- 熟练掌握 Python 语言
- 熟悉 PaddleSpeech 框架及其数据处理流程

### NO.7 PaddleSpeech 新 Python 版本适配

**详细描述：**

- 适配 python3.9-3.12 版本，梳理套件依赖库

**验收标准：**

- 尽可能清理依赖库，有风险的依赖库固定版本
- 能够在多个 python 版本上使用 pip 成功安装 paddlespeech
- 安装成功后 demo 和 example 目录下的模型能够正常跑通

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 PaddleSpeech 框架及其数据处理流程

### NO.8

**详细描述：**

- 在 PaddleSpeech 套件中实现并对齐 CosyVoice 的组网和模型推理
- 参考：https://github.com/FunAudioLLM/CosyVoice

**验收标准**：

- 复现的性能指标需要与原始仓库效果相当
- 需上传完整的训练代码和训练脚本
- 训练后的模型支持 command line 方式推理
- 代码相关文档和注释完备

**技术要求：**

- 了解 Whisper 模型
- 熟练掌握 Python 语言
- 熟悉 PaddleSpeech 框架及其数据处理流程

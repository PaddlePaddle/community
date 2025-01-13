此文档展示 **PaddlePaddle Hackathon 第八期活动——开源贡献个人挑战赛套件开发方向任务** 详细介绍

## 【开源贡献个人挑战赛-套件开发】任务详情

### NO.5 在 PaddleSpeech 中复现 DAC 模型 (依赖任务 NO.9)

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

### NO.8 在 PaddleSpeech 中实现 CosyVoice 模型推理

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

### NO.9 在 PaddleSpeech 中复现 DAC 的训练需要用到的 loss

**详细描述：**

- 在 PaddleSpeech 套件中实现并对齐 Descript-Audio-Codec 中使用到的 MultiScaleSTFTLoss，GANLoss，SISDRLoss。
- 相关论文：https://arxiv.org/abs/2306.06546
- 参考：https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
- 相关实现放在：paddlespeech/t2s/modules/losses.py

**验收标准**：

- 复现的精度需要与原 repo 保持一致

**技术要求：**

- 熟练掌握 Python 语言

## 【开源贡献个人挑战赛-科学计算】任务详情

### 开发流程

1. **要求基于 PaddleScience 套件进行开发**，开发文档参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ 。
2. 复现整体流程和验收标准可以参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#21，复现完成后需供必要的训练产物，包括训练结束后保存的 `train.log`日志文件、`.pdparams`模型权重参数文件（可用网盘的方式提交）、**撰写的 `.md` 案例文档。**
3. 理解复现流程后，可以参考 PaddleScience 开发文档：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ ，了解各个模块如何进行开发、修改，以及参考 API 文档，了解各个现有 API 的功能和作用：https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/ 。
4. 案例文档撰写格式可参考 https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d/ ，最终合入后会被渲染并展示在 [PaddleScience 官网文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/)。
5. **如在复现过程中出现需添加的功能无法兼容现有 PaddleScience API 体系（[PaddleScience API 文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/)），则可与论文复现指导人说明情况，并视情况允许直接基于 Paddle API 进行复现。**
6. 若参考代码为 pytorch，则复现过程可以尝试使用 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 辅助完成代码转换工作，然后可以尝试使用 [PaDiff](https://github.com/PaddlePaddle/PaDiff) 工具辅助完成前反向精度对齐，从而提高复现效率。

### 验收标准

参考模型复现指南验收标准部分 https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#3

## 【开源贡献个人挑战赛-科学计算方向】任务详情

### NO.10 Transolver 论文复现

**论文链接：**

https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=Transolver%3A+A+Fast+Transformer+Solver+for+PDEs+on+General+Geometries&btnG=#:~:text=%E5%8C%85%E5%90%AB%E5%BC%95%E7%94%A8-,%5BPDF%5D%20arxiv.org,-Transolver%3A%20A%20fast

**代码复现：**

复现 List of experiments:

- Core code: see [./Physics_Attention.py](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
- Standard benchmarks: see [./PDE-Solving-StandardBenchmark](https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark)
- Car design task: see [./Car-Design-ShapeNetCar](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar)
- Airfoil design task: see [./Airfoil-Design-AirfRANS](https://github.com/thuml/Transolver/tree/main/Airfoil-Design-AirfRANS)

精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/thuml/Transolver

---

### NO.11 DrivAerNet ++ 论文复现

**论文链接：**

https://github.com/Mohamedelrefaie/DrivAerNet#:~:text=preprint%3A%20DrivAerNet%2B%2B%20paper-,here,-DrivAerNet%20Paper%3A

**代码复现：**

复现 RegDGCNN 和 PointNet，在数据集 DrivAer++上，精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/Mohamedelrefaie/DrivAerNet

---  

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

- 了解 CosyVoice 模型
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

### NO.28 Phi3 模型复现

**详细描述：**

​ 在 PaddleNLP 套件中实现并对齐 Phi3 的组网

- 参考：https://huggingface.co/microsoft/phi-4
- 参考：https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi3
- 精度验证方法：https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/align_pytorch_and_paddle.html

**验收标准：**

- 复现的精度需要与原始代码接近，在 fp32 下误差小于 1e-5，bf16 下误差小于 1e-3
- 在 PaddleNLP 套件中实现模型组网和 tokenizer
- 在 PaddleNLP 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉大模型结构

### NO.29 ModernBert 模型复现

**详细描述：**

在 PaddleNLP 套件中实现并对齐 ModernBert 的组网

- 参考：https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert
- 精度验证方法：https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/align_pytorch_and_paddle.html

**验收标准：**

- 复现的精度需要与原始代码接近，在 fp32 下误差小于 1e-5，bf16 下误差小于 1e-3
- 在 PaddleNLP 套件中实现模型组网和 tokenizer
- 在 PaddleNLP 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉大模型结构

### NO.30 Gemma2 模型复现

**详细描述：**

在 PaddleNLP 套件中实现并对齐 Gemma2 的组网

- 参考：https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma2
- 精度验证方法：https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/align_pytorch_and_paddle.html

**验收标准：**

- 复现的精度需要与原始代码接近，在 fp32 下误差小于 1e-5，bf16 下误差小于 1e-3
- 在 PaddleNLP 套件中实现模型组网和 tokenizer
- 在 PaddleNLP 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉大模型结构

### NO.31 Apollo 精调算法复现

**详细描述：**

在 PaddleNLP 套件中实现 Apollo 优化器

- 参考开源代码：https://github.com/zhuhanqing/APOLLO
- 参考论文：https://arxiv.org/pdf/2412.05270

**验收标准：**

- 复现论文 5.2 节中 Apollo 与 LoRA 在 Llama3-8B 上的实验结果，优化器状态显存占用相比 AdamW 降低 50% 以上
- 代码相关文档和注释完备，且代码符合规范

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 LoRA 等主流 PEFT 方法

### NO.32 Adam-mini 精调算法复现

**详细描述：**

在 PaddleNLP 套件中实现分块版本的 Adam-mini 优化器

- 参考开源代码：https://github.com/zyushun/Adam-mini
- 参考 PaddleNLP 已实现的未分块版本的 adam-mini：https://github.com/PaddlePaddle/PaddleNLP/blob/55db2ff0673193a59cac37bfc917b1f2a67646af/paddlenlp/utils/optimizer.py#L24
- 参考论文：https://arxiv.org/pdf/2406.16793

**验收标准：**

- 复现论文 3.1 节中 Adam-mini 在 Llama3-8B 上的实验结果，优化器状态显存占用相比 AdamW 降低 50%
- 代码相关文档和注释完备，且代码符合规范

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉 LoRA 等主流 PEFT 方法

### NO.33 PaddleNLP CI 覆盖模型测试

**详细描述：**

- 在 PaddleNLP 套件中实现不同模型的流水线监控，包括 [Yuan2](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/tests/transformers/yuan)、[llm_embed](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/tests/transformers/llm_embed)、[DeepSeekV2/V3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/tests/transformers/deepseek_v2)，参考开源代码 Llama 和 Qwen2 相关实现，[链接](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/tests/transformers/qwen2)。
- 实现Yuan2和DeepSeekV2模型在[llm](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/tests/llm)上的验证，包括[Pretrain](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/tests/llm/test_pretrain.py)、[Fintune](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/tests/llm/test_finetune.py)、[Lora](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/tests/llm/test_lora.py)、[Predictor](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/tests/llm/test_predictor.py)等阶段验证。
- 在验证过程中，如需协助上传相关文件，可联系研发同学。

**验收标准：**

- 保证 CI 任务执行成功

**技术要求：**

- 熟练掌握 Python Shell 语言

### NO.34 Flux ControlNet 模型复现

**详细描述：**

在 PaddleMIX 套件中实现并对齐 Flux ControlNet 的组网和模型推理

- 参考开源代码：

  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py
  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/controlnet_flux.py

- 参考相关 PR：https://github.com/PaddlePaddle/PaddleMIX/pull/829

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleMIX 套件 PPDiffusers 模块中实现模型组网和相关推理 Pipeline
- 在 PaddleMIX 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉扩散模型原理

### NO.35 MOchi-1 模型复现

**详细描述：**

在 PaddleMIX 套件中实现并对齐 Mochi-1 的组网和模型推理

- 参考开源代码：

  - https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/mochi
  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_mochi.py

- 参考相关 PR：https://github.com/PaddlePaddle/PaddleMIX/pull/663

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleMIX 套件 PPDiffusers 模块中实现模型组网和相关推理 Pipeline
- 在 PaddleMIX 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉扩散模型原理

### NO.36 gme-Qwen2-VL-2B-Instruct 模型复现

**详细描述：**

在 PaddleMIX 套件中实现并对齐 gme-Qwen2-VL-2B-Instruct 的组网和模型推理

- 参考开源代码：https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py
- 参考相关 PR：https://github.com/PaddlePaddle/PaddleMIX/pull/1038

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleMIX 套件 paddlemix/models 和 paddlemix/examples 模块中实现模型组网和相关推理 Pipeline
- 在 PaddleMIX 套件中新增对应单测
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言

- 熟悉扩散模型原理

### NO.37 YOLOv11 模型复现

**详细描述：**

在 PaddleYOLO 套件中实现并对齐 YOLOv11 的组网

- 参考开源代码：https://github.com/ultralytics/ultralytics
- YOLOv11 相关配置：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml（可参考套件内已实现的 YOLO 系列算法）

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleYOLO 套件中实现模型组网
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语
- 熟悉检测模型原理

### NO.38 RT-DETRv3 模型复现

**详细描述：**

在 PaddleDetection 套件中实现并对齐 RT-DETRv3 的组网

- 参考开源代码：https://github.com/clxia12/RT-DETRv3 （可参考已实现的 RT-DETR 和 RT-DETRv2：https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/rtdetrv2 ）

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleDetection 套件中实现模型组网
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉检测模型原理

### NO.39 DEIM 模型复现

**详细描述：**

- 在 PaddleDetection 套件中实现并对齐 DEIM 的组网
- 参考开源代码：DEIM（[DETR with Improved Matching for Fast Convergence](https://arxiv.org/abs/2412.04234)，可参考套件内其他检测模型）

**验收标准：**

- 复现的性能指标需要与原始仓库效果相当
- 在 PaddleDetection 套件中实现模型组网
- 代码相关文档和注释完备

**技术要求：**

- 熟练掌握 Python 语言
- 熟悉检测模型原理

## 【开源贡献个人挑战赛-科学计算方向】任务详情

#### 开发流程

1. **要求基于 PaddleScience 套件进行开发**，开发文档参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ 。
2. 复现整体流程和验收标准可以参考：https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#21，复现完成后需供必要的训练产物，包括训练结束后保存的 `train.log`日志文件、`.pdparams`模型权重参数文件（可用网盘的方式提交）、**撰写的 `.md` 案例文档。**
3. 理解复现流程后，可以参考 PaddleScience 开发文档：https://paddlescience-docs.readthedocs.io/zh/latest/zh/development/ ，了解各个模块如何进行开发、修改，以及参考 API 文档，了解各个现有 API 的功能和作用：https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/ 。
4. 案例文档撰写格式可参考 https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/darcy2d/ ，最终合入后会被渲染并展示在 [PaddleScience 官网文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/)。
5. **如在复现过程中出现需添加的功能无法兼容现有 PaddleScience API 体系（[PaddleScience API 文档](https://paddlescience-docs.readthedocs.io/zh/latest/zh/api/arch/)），则可与论文复现指导人说明情况，并视情况允许直接基于 Paddle API 进行复现。**
6. 若参考代码为 pytorch，则复现过程可以尝试使用 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 辅助完成代码转换工作，然后可以尝试使用 [PaDiff](https://github.com/PaddlePaddle/PaDiff) 工具辅助完成前反向精度对齐，从而提高复现效率。

#### 验收标准

参考模型复现指南验收标准部分 https://paddlescience-docs.readthedocs.io/zh/latest/zh/reproduction/#3

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

### NO.11 DrivAerNet ++ 论文复现

**论文链接：**

https://github.com/Mohamedelrefaie/DrivAerNet#:~:text=preprint%3A%20DrivAerNet%2B%2B%20paper-,here,-DrivAerNet%20Paper%3A

**代码复现：**

复现 RegDGCNN 和 PointNet，在数据集 DrivAer++上，精度与论文中对齐，完成文档，符合代码审核要求

**参考代码链接：**

https://github.com/Mohamedelrefaie/DrivAerNet

### NO.12 在 PaddleScience 中实现 SOAP 优化器

**论文链接：**

https://arxiv.org/abs/2502.00604

**代码复现：**

基于 Allen-Cahn + SOAP 进行复现，达到论文中对应的精度，优化器合入 Paddle 和 PaddleScience 仓库

**参考代码链接：**

https://github.com/haydn-jones/SOAP_JAX/tree/main

https://github.com/nikhilvyas/SOAP

### NO.13 Domino 论文复现

**论文链接：**

https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/domino

**代码复现：**

复现 [Domino](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/domino) 模型推理，精度与论文中对齐，并合入 PaddleScience

### NO.14 CoNFiLD 论文复现

**论文链接：**

https://github.com/jx-wang-s-group/CoNFiLD

**代码复现：**

复现 [CoNFiLD](https://github.com/jx-wang-s-group/CoNFiLD) 模型推理，精度与论文中对齐，并合入 PaddleScience

### NO.15 Diffeomorphism Neural Operator 论文复现

**论文链接：**

https://github.com/Zhaozhiwhy/Diffeomorphism-Neural-Operator

**代码复现：**

复现 [Diffeomorphism-Neural-Operator](https://github.com/Zhaozhiwhy/Diffeomorphism-Neural-Operator) 模型推理，精度与论文中对齐，并合入 PaddleScience

### NO.16 Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning 论文复现

**论文链接：**

https://github.com/delta-lab-ai/data_efficient_nopt

**代码复现：**

复现 [data_efficient_nopt](https://github.com/delta-lab-ai/data_efficient_nopt) 模型推理，精度与论文中对齐，并合入 PaddleScience

### NO.17 FuXi 论文复现

**论文链接：**

https://arxiv.org/abs/2306.12873

**代码复现：**

复现 FuXi 模型推理，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/tpys/FuXi

### NO.18 FengWu 论文复现

**论文链接：**

https://arxiv.org/abs/2304.02948

**代码复现：**

复现 FengWu 模型推理，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/OpenEarthLab/FengWu

### NO.19 Pangu-Weather 论文复现

**论文链接：**

https://arxiv.org/abs/2211.02556

**代码复现：**

复现 Pangu-Weather 模型推理，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://github.com/198808xc/Pangu-Weather

### NO.20 PDEformer-1 论文复现

**论文链接：**

https://arxiv.org/abs/2407.06664

**代码复现：**

复现 PDEformer 模型，精度与论文中对齐，并合入 PaddleScience

**参考代码链接：**

https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/pdeformer1d

### NO.21 Synthetic Lagrangian turbulence by generative diffusion models 论文复现

**论文链接：**

https://www.nature.com/articles/s42256-024-00810-0

**代码复现：**

复现模型及案例，精度与论文中对齐，并合入 PaddleScience 仓库

**参考代码链接：**

https://github.com/SmartTURB/diffusion-lagr

### NO.22 From Zero to Turbulence 论文复现

**论文链接：**

https://openreview.net/forum?id=ZhlwoC1XaN

**代码复现：**

复现模型及三维湍流模拟案例，精度与论文中对齐，并合入 PaddleScience 仓库

**参考代码链接：**

https://github.com/martenlienen/generative-turbulence

### NO.23 Improved Training of Wasserstein GANs 论文复现

**论文链接：**

https://arxiv.org/abs/1704.00028

**代码复现：**

复现 WGAN-GP 模型及其案例，精度与论文中对齐，并合入 PaddleScience 仓库

**参考代码链接：**

https://github.com/igul222/improved_wgan_training

### NO.24 MatterSim 论文复现

**论文链接：**

https://arxiv.org/abs/2405.04967

**代码复现：**

复现高影响力 MatterSim 模型，实现推理功能和微调功能，精度与参考代码对齐

**参考代码链接：**

https://github.com/microsoft/mattersim

### NO.25 PotNet 论文复现

**论文链接：**

https://arxiv.org/abs/2306.10045

**代码复现：**

复现高影响力 PotNet 模型，实现训练、验证、推理等功能，精度与参考代码对齐

**参考代码链接：**

https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet

### NO.26 Matformer 论文复现

**论文链接：**

https://arxiv.org/abs/2209.11807

**代码复现：**

复现高影响力 Matformer 模型，实现训练、验证、推理等功能，精度与参考代码对齐

**参考代码链接：**

https://github.com/divelab/AIRS/tree/main/OpenMat/Matformer

### NO.27 EquiCSP 论文复现

**论文链接：**

https://proceedings.mlr.press/v235/lin24b.html

**代码复现：**

复现基于 Diffusion 的 EquiCSP 模型，实现训练、验证、结构生成等功能，精度与参考代码对齐

**参考代码链接：**

https://github.com/EmperorJia/EquiCSP

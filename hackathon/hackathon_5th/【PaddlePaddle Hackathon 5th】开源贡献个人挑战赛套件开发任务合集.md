此文档展示 **PaddlePaddle Hackathon 第五期活动——开源贡献个人挑战赛套件开发任务** 详细介绍，更多详见  [PaddlePaddle Hackathon 说明](https://www.paddlepaddle.org.cn/contributionguide?docPath=hackathon_cn)。

## 【开源贡献个人挑战赛-套件开发】任务详情

### No.64：全套件模型接入动转静训练功能

**任务背景：**

目前飞桨的开源套件如PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR等，都支持了动转静训练功能，但是并非所有的模型都接入了`--to_static`策略，随着PaddleSOT 功能的完善和上线，动转静训练成功率大幅度提升，故此挑战赛旨在对开源套件中所有模型进行动转静训练策略推全。本题完成一个方向即可获得1颗🌟，一共可获得6颗🌟。

**详细描述：**

任务需要同学对现有的PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR套件中的所有模型依次添加 to static 策略，支持开启动转静进行训练，且保证对套件模型尽可能少的代码侵入。具体包含如下阶段：

- 明确全套件列表，包含：PaddelClas、PaddleNLP、PaddleSeg、PaddleDetection、Paddle3D、PaddleOCR。
- 在每个套件中，同学需要调研或者询问套件负责人，搜集套件的所有模型列表，并对所有模型的动转静支持情况进行调研。**产出《待支持动转静模型列表文档》。**
- 针对每个待支持动转静的模型，对套件代码进行修改，以支持动转静训练。同时提供开启动转静训练前后前50个step的loss一致性截图作为PR描述，[样例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/1290/files)。
- 让熊昆和套件负责人review，同意并合入PR后，此模型视为接入动转静。

**提交内容：**

- **提交一份《待支持动转静模型列表文档》作为进度验收表**
- 针对每个待支持模型，提供PR并且提供开启动转静训练前后前50个step的loss一致性截图作为PR描述。
- **完成一个方向所有模型的动转静即计为完成一次。**

**技术要求：**

- 熟练掌握 C++，Python。
- 了解如何运行和修改套件代码。
- 了解如何启动动转静训练。

### No.65：版面恢复功能（恢复为docx或者excel）的c++版

**任务背景：**

用于手机端的本地化办公文档扫描。

**详细描述：**

1. 版面分析和OCR后，使用minidocx创建docx文档，libxlsxwriter生成excel。
2. 在[ppstruncture的推理代码](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/paddlestructure.cpp)上增加相关功能，提交PR到PaddleOCR。

**提交内容：**

按照PR提交规范，提交支持c++生成文档的代码，并展示部署效果。

### No.66：生僻词模型训练

**任务背景：**

OCR的中文字符目前存在字典不全问题，没有覆盖《通用规范汉字表》；对于字典中存在的生僻字，可能因为训练语料不平衡问题，识别效果很差。新增生僻字模型能大幅提升模型在身份证、古文识别场景的能力。

**详细描述：**

1. 替换现有字典txt为扩充《通用规范汉字表》的字典。
2. 在现有数据集（**任意公开数据集均可**）上通过数据合成copy paste等方式实现语料的平衡，并重新训练PPOCRV4的检测和识别模型。
3. 对比训练后模型在普通文字和生僻字上的检测、识别精度，并和PPOCRV4模型最优模型进行对比；达到普通字精度不变或者更高，生僻字上精度进一步提升的效果。
4. 提交PR到ppocr，替换最优模型。

**提交内容：**

提交训练后的模型链接到ppocr，并提供readme展示对比效果。

### No.67：版面矫正网络DocTr++论文复现

**任务背景：**

DocTr++版面矫正在文档比对、关键字提取、合同篡改确认等重要场景发挥作用。本任务的完成能显著OCR结果的细粒度，并有众多场景应用。
通过定量实验和定性对比，作者团队验证了 DocTr++ 的性能优势及泛化性，并在现有及所提出的基准测试中刷新了多项最佳记录，是目前最优的文档矫正方案。

暂时没有预训练权重和训练代码，需要按照论文描述重新训练尝试。

**详细描述：**

1. 根据开源代码进行网络结构、评估指标转换，[代码链接](https://github.com/fh2019ustc/DocTr-Plus)。
2. 结合[论文复现指南](https://github.com/PaddlePaddle/models/blob/release/2.4/tutorials/article-implementation/ArticleReproduction_CV.md)，进行前反向对齐等操作，达到论文Table.1中的指标。
3. 参考[PR提交规范](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/code_and_doc.md)提交代码PR到ppocr中。

**数据集：**

1. 训练数据集：获取[Doc3D数据集](https://github.com/cvlab-stonybrook/doc3D-dataset)后进行边缘裁剪，使得分成论文中的三类图片（全部包含边缘、部分包含边缘、不包含边缘），详细情况参考论文训练集中描述。
2. 验证数据集：[Doc Unet数据集](https://www3.cs.stonybrook.edu/~cvl/docunet.html)。

**提交内容：**

正确进行版面矫正的模型代码和权重，提交PR到PaddleOCR。

### No.68：轻量语义分割网络PIDNet

**任务背景：**

该模型为轻量化分割方向的前沿模型，超过自研模型ppliteseg精度和速度平衡，Cityscapes上精度直逼高精度OCRNet，数据和模型、代码均已经开源。

**详细描述：**

1. 数据和模型、代码均已经开源。
2. 根据开源代码进行网络结构、评估指标转换，[代码链接](https://github.com/XuJiacong/PIDNet)。
3. 结合[论文复现指南](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/article-implementation/ArticleReproduction_CV.md)，进行前反向对齐等操作，达到论文Table.6中的指标。
4. 进行TIPC验证lite train lite infer 链条。
5. 参考[PR提交规范](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/pr/pr/style_cn.md)提交代码PR到[ppseg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)中。

**提交内容：**

1. 代码提交到PaddleSeg。

### No.69：分类大模型--人体视觉任务SOLIDER

**任务背景：**

该论文利用自监督训练方式，充分利用现有大量人体无标注数据，得到一个可以通用于下游各种人体视觉任务的预训练大模型，本任务的完成可以支持PaddleClas各种人体视觉任务。
现已有开源代码，该论文只需前向对齐即可，即输入相同图片，输出结果差距在1e-6以内。

**详细描述：**

1. 根据开源代码进行网络结构转换，[代码链接](https://github.com/tinyvision/SOLIDER)。
2. 参考[论文复现指南](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/article-implementation/ArticleReproduction_CV.md)，本任务只需要对齐前向，无需训练，即使用[demo.py](https://github.com/tinyvision/SOLIDER/blob/main/demo.py)来前向对齐，需对齐的模型包括swin_tiny_patch4_window7_224、swin_small_patch4_window7_224以及swin_base_patch4_window7_224，因PaddleClas已有这些模型，只需在现有模型进行修改，无需创建新的模型代码。
3. 参考[PR提交规范](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/community/how_to_contribute.md)提交代码PR到PaddleClas中。

**提交内容：**

1. 增加介绍文档PaddleClas/docs/zh_CN/models/sodier.md
2. 对swin系列backbone进行必要的修改
3. 发送转化swin系列（swin_tiny_patch4_window7_224、swin_small_patch4_window7_224以及swin_base_patch4_window7_224）的权重和对齐日志。

### No.70：DET重点模型支持实例分割

**任务背景：**

实例分割使用场景比较广泛，目前PaddleDetection支持的实例分割模型较老，不能满足用户需求，需要支持。

**详细描述：**

1. 对PP-YOLO-E+_crn_l、RT-DETR-L模型新增实例分割头，且在COCO数据集上达到较同样level模型的更高的精度。
2. 打通基于python的部署，文档齐全。

### No.71：新增 bevfusion 部署链条 

**任务背景：**

该任务基于Paddle Inference为bevfusion增加python和C++的部署链条，为该3D模型的部署助力。

**详细描述：**

1. 下载[动态图模型](https://github.com/PaddlePaddle/Paddle3D/tree/develop/docs/models/bevfusion)，进行静态图导出
2. 基于导出的模型进行python链条的部署和C++部署的验证，代码结构可以参考[PETR推理部署](https://github.com/PaddlePaddle/Paddle3D/tree/develop/deploy/petr)。

**提交内容：**

参照其他部署文件，提交部署内容到[Paddle3D/develop/deploy/](https://github.com/PaddlePaddle/Paddle3D/tree/develop/deploy)bevfusion文件夹下。

### No.72：新增模型TaskMatrix 

**任务背景：**

该模型建立了一个VIsual ChatGPT系统，实现了对任意图片进行视觉编辑和图文问答。该算法代码已经开源，需要调用paddlemix和paddlenlp中已经集成的模型，接入llm进行视觉对话系统的搭建。该算法不需要进行模型转换等，只需要对现有模型进行串联。

**详细描述：**

1. 实现visualChatGPT，并进行相应验证，代码链接。https://github.com/microsoft/TaskMatrix
2. 接入开源模型例如chatglm v2或者llama v2，来实现中文版本的Visual ChatGPT，给出使用示例和文档以及UI。提交至https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/agents

**提交内容：**

1. 实现使用示例和文档以及UI。
2. 提交代码和readme到：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/agents

### No.73：新增模型Tree of Thoughts

**任务背景：**

ToT提出了一种新的思维架构来显著提升GPT解决问题的能力，通过考虑多个不同的推理路径和自我评估来提升行动的成功率。例如，在《24点游戏》中，具有思维链提示的GPT-4只解决了4%的任务，而ToT的成功率为74%。该算法代码已开源，需要将代码转换并接入开源语言模型。

**详细描述：**

1. 仿照ReAct的方式集成到pipelines里面，并评估跟论文精度一致。论文链接：https://github.com/princeton-nlp/tree-of-thought-llm
2. 接入开源模型例如chatglm v2或者llama v2，并给出TOT的使用示例和文档。提交至https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/agents

**提交内容：**

提交上文描述的代码和readme到：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/agents

### No.74：RetroMAE训练

**任务背景：**

RetroMAE，这是一种基于掩码自动编码器（MAE）的新的面向检索的预训练范式。RetroMAE有三个关键设计。1） 一种新的MAE工作流程，其中输入句子被不同掩码的编码器和解码器污染。句子嵌入是从编码器的屏蔽输入生成的；然后，通过掩蔽语言建模，基于句子嵌入和解码器的掩蔽输入来恢复原始句子。2） 非对称模型结构，以全尺寸类BERT变换器作为编码器，以单层变换器作为解码器。3） 不对称掩蔽率，编码器的掩蔽率适中：15-30%，解码器的掩蔽率激进：50-70%。预训练的模型在广泛的密集检索基准上显著提高了SOTA的性能，如BEIR和MS MARCO。

paper: https://arxiv.org/abs/2205.12035

code: https://github.com/FlagOpen/FlagEmbedding

**详细描述：**

1.BGE中英文模型前向对齐。

2.用Trainer的方式实现BGE Embedding训练，并对齐。

**提交内容：**

将对齐的代码提交至：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/semantic_indexing

### No.75：新增模型InstructBlip

**任务背景：**

InstructBlip是基于blip2的一种tuning的结构。

**详细描述：**

- 论文：https://arxiv.org/abs/2305.06500
- 代码：[https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS/tree/main/lavis/datasets/datasets)
- 模型选取 opt2.7b即可
- 模型结构和竞品对齐，前向 + 反向 + 评估，对齐论文Table 1
- https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
- **demo结果输出与 Figure 5 一致。**

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.76：新增数据集训练和评估 (coco retrieval)

**任务背景：**

完善数据的训练和评估，[参考资料](https://github.com/salesforce/LAVIS/tree/main/lavis/datasets/datasets)

**详细描述：**

- 需要在coco retrieval 标准数据集上训练 + 评估出指标
- 在blip2上进行的评估，对齐论文呢table1的结果。

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.77：新增模型kosmos2

**任务背景：**

大模型做感知任务，refering, grounding等。[论文](https://arxiv.org/abs/2306.14824)，[代码](https://github.com/microsoft/unilm)

**详细描述：**

- 需要在标准数据集上训练 + 评估出指标，对齐论文tabel3的结果。（可以权重转过来评估，训练前反向对齐能跑通即可）
- 论文中demo对齐Figure 10。

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.78：minigpt-4 zeroshot评估

**任务背景：**

minigpt-4量化能力评估，[代码](https://github.com/salesforce/LAVIS/tree/main/lavis)

**详细描述：**

- 设计评估的方案，可人工评估和标准数据集评估
- 例如
  - 在coco 上进行cap, vqa, retrieval评估
- **论文中无具体指标，跑通即可 指标相对合理即可 开放性题目**

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.79：新增模型openseed

**任务背景：**

[openseed](https://github.com/IDEA-Research/OpenSeeD)是一个Open Vocabulary Learning方向的[算法](https://arxiv.org/pdf/2303.08131.pdf)，一个模型完成检测和分割功能。

**详细描述：**

需要在标准数据集上训练 + 评估出指标，对齐论文table3中结果55.4PQ

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.80：添加appflow以及对应模型单测

**任务背景：**

[appflow](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/README.md)是paddlemix实现跨模态多场景应用的统一接口，串联多个模型与pipeline。

**详细描述：**

- 将已接入appflow的应用以及对应的模型添加单测，应用+模型数量：12个
- 参考https://github.com/PaddlePaddle/PaddleMIX/blob/develop/tests/models/test_blip2.py ，添加groudingdino、sam模型的单测：2个，
- 添加应用单测：10个 
  - [自动标注（AutoLabel）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/Automatic_label/README.md/#自动标注autolabel)
  - [文图生成（Text-to-Image Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/text2image/README.md/#文图生成text-to-image-generation)
  - [文本引导的图像放大（Text-Guided Image Upscaling）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本引导的图像放大text-guided-image-upscaling)
  - [文本引导的图像编辑（Text-Guided Image Inpainting）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/Inpainting/README.md/#文本引导的图像编辑text-guided-image-inpainting)
  - [文本引导的图像变换（Image-to-Image Text-Guided Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本引导的图像变换image-to-image-text-guided-generation)
  - [文本图像双引导图像生成（Dual Text and Image Guided Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本图像双引导图像生成dual-text-and-image-guided-generation)
  - [文本条件的视频生成（Text-to-Video Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/text2video/README.md/#文本条件的视频生成text-to-video-generation)
  - [音频描述（Audio-to-Caption Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/Audio2Caption/README.md/#音频描述audio-to-caption-generation)
  - [音频对话（Audio-to-Chat Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/AudioChat/README.md/#音频对话audio-to-chat-generation)
  - [音乐生成（Music Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/MusicGeneration/README.md/#音乐生成music-generation)

- 单测示例： https://github.com/PaddlePaddle/PaddleMIX/blob/develop/tests/appflow/test_cviw.py

**提交内容：**

- 2+11个单测脚本

- 提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.81：applications应用gradio demo

**任务背景：**

paddlemix基于[appflow api](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/README.md)实现多个应用,每类应用需要gradio demo

**详细描述：**

- 将已接入appflow的2个应用添加gradio demo。
  - [自动标注（AutoLabel）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/Automatic_label/README.md/#自动标注autolabel) ：
    - 要求输出labelme格式的json，如 [示例.json](./示例.json)
    - 支持批量输入输出
- text2image ，包含以下功能：
  - [文本引导的图像放大（Text-Guided Image Upscaling）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本引导的图像放大text-guided-image-upscaling)
  - [文本引导的图像变换（Image-to-Image Text-Guided Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本引导的图像变换image-to-image-text-guided-generation)
  - [文本图像双引导图像生成（Dual Text and Image Guided Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/image2image/README.md/#文本图像双引导图像生成dual-text-and-image-guided-generation)
  - [文本条件的视频生成（Text-to-Video Generation）](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/applications/text2video/README.md/#文本条件的视频生成text-to-video-generation)
- 示例： https://github.com/LokeZhou/PaddleMIX/blob/gradio/applications/gradio/chat_inpainting_gradio.py

**提交内容：**

提交2个gradio脚本，分别是gradio_autolable.py；gradio_text2image.py

- 提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.82：为Paddle2ONNX增加原生FP6 Paddle模型的转换能力

**任务背景：**

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

目前Paddle2ONNX对现有的PaddlePaddle Frontend覆盖算子不够全面，覆盖度仅包含各个套间(PaddleOCR/PaddleDetection等)的部分模型，我们希望通过此次黑客松活动为Paddle2ONNX中的PaddlePaddle Frontend补充关于FP16算子的支持。

**目标：**

为Paddle2ONNX添加FP16支持，需要同时满足以下要求:

- 要求能够将ResNet-50(FP16)成功转换为ONNX模型
- 要求对齐ResNet-50(FP16)的精度且误差不超过0.1%，如果不能达到需要在提交的PR中说明原因。

**注意事项：**

- 任务给定的ResNet-50(FP16)模型将在后续放出
- 请注意代码风格和Paddle2ONNX保持一致
- OP代码以及单测实现之后，@jiangjiajun进行code review以及代码修改，修改完成后即可代码合入

**可能用到的链接：**

- [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)

### No.83：PaddleMIX ppdiffusers models模块功能升级同步HF

**任务背景：**

PaddleMIX ppdiffusers 作为飞桨扩散模型基础设施其目标之一是为飞桨开发者提供社区最新的扩散模型能力支持，开源社区目前一些新的能力需要进一步补充跟进。

**详细描述：**

基于ppdiffusers最新代码完成向HF新版本能力的扩充，主要完成 https://github.com/huggingface/diffusers/compare/v0.19.3...v0.21.1 models 模块升级同步

- 保证功能覆盖
- 保证单测

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop



### No.84：新增模型视频生成模型MS-Image2Video+MS-Vid2Vid-XL

**任务背景：**
视频生成是当下继图片生成之后AIGC的另一热点，相关工作愈加成熟，希望进一步完善飞桨在视频生成领域的能力。

**详细描述：**
* 基于PaddleMIX ppdiffusers完成MS-Image2Video和MS-Vid2Vid-XL的模型转换和前向对齐；转换模型权重进行生成，保证生成效果的对齐；
* 将对齐的模型应用到pipeline，参考 https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/inference/text_to_video_generation-synth.py；
* 生成提供gradio demo用于验证串联效果。

**提交内容：**
提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop


### No.85：新增虚拟试衣模型应用 DCI-VTON-Virtual-Try-On
**任务背景：**
虚拟试衣场景任务具有一定的研究和应用价值，希望补充飞桨在该场景任务上的能力。

**详细描述：**
* 基于PaddleMIX ppdiffusers完成DI-VTON-Virtual-Try-On模型前向对齐，定量指标对齐[Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow](https://github.com/bcmi/DCI-VTON-Virtual-Try-On#dci-vton-virtual-try-on)的Table 1
* 将对齐的模型应用到pipeline，参考 https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/inference/text_to_video_generation-synth.py；
* 实现训练功能，训练Warping Module和Diffusion Model评估对齐原repo。

**提交内容：**
提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop



### No.86：新增图像组合模型应用TF-ICON
**任务背景：**
图像组合尤其是不同domain的图像组合具有一定的应用价值，希望新增tuning free的cross-domain image-guided composition场景任务能力。

**详细描述：**
* 基于PaddleMIX ppdiffusers完成TF-ICON模型的前向对齐，使用SD（sd-v2-1_512-ema-pruned）模型权重和TF-ICON Test Benchmark上保证Image Composition的生成效果，且定量指标对齐[TF-ICON: Diffusion-Based Training-Free Cross-Domain Image Composition](https://shilin-lu.github.io/tf-icon.github.io/)的Table 3
和应用pipeline，参考https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/community 下的pipeline实现；
* 提供gradio demo验证pipeline效果。

**提交内容：**
提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

### No.87：PaddleMIX ppdiffusers新增HF community应用pipeline

**任务背景：**

HF提供了丰富的应用pipeline，其中部分具有一定的应用价值，希望补齐这些应用能力，为飞桨开发者提供社区最新的扩散模型能力支持。

**详细描述：**

基于PaddleMIX ppdiffusers完成以下pipeline，保证生成效果对齐

edict_pipeline.py  https://github.com/huggingface/diffusers/blob/main/examples/community/edict_pipeline.py

pipeline_fabric.py https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_fabric.py

**提交内容：**

提交到https://github.com/PaddlePaddle/PaddleMIX/tree/develop

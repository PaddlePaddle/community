# 【PaddlePaddle Hackathon 4】模型套件开源贡献任务合集

（此 ISSUE 为 PaddlePaddle Hackathon 第四期活动的任务 ISSUE，更多详见 [【PaddlePaddle Hackathon 第四期】任务总览](https://github.com/PaddlePaddle/Paddle/issues/50629)）

注：为飞桨框架新增一系列 API，提交流程请参考 [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)，开发请参考 [贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)，任务列表如下，其他说明事项在任务列表后：

### No.98：升级paddlenlp.transformers内的模型结构并且增加基础单测 <a name='task98'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：**基础⭐️
- **详细描述：**
  - 升级指定的模型PaddleNLP模型结构，每个模型的主要工作为：
    - 为模型结构增加configuration.py, 对齐huggingface/transformers的config，并且适配在模型代码中适配config, 详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
    - 为模型增加单测, 并且做到单测通过，详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
  - 每个模型算单独的子任务，**每升级3个模型算完成一个基础任务**, 总共待升级的模型为
    - [MegatronBERT](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/megatronbert)
    - [artist](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/artist)
    - [dallebart](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/dallebart)
  - 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)

- **提交内容：**
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法


### No.99：升级paddlenlp.transformers内的模型结构并且增加基础单测 <a name='task99'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：**基础⭐️
- **详细描述：**
  - 升级指定的模型PaddleNLP模型结构，每个模型的主要工作为：
    - 为模型结构增加configuration.py, 对齐huggingface/transformers的config，并且适配在模型代码中适配config, 详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
    - 为模型增加单测, 并且做到单测通过，详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
  - 每个模型算单独的子任务，**每升级3个模型算完成一个基础任务**, 总共待升级的模型为
    - [funnel](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/funnel)
    - [fnet](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/fnet)
    - [mpnet](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/mpnet)
  - 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)

- **提交内容：**
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法


### No.100：升级paddlenlp.transformers内的模型结构并且增加基础单测 <a name='task100'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：**基础⭐️
- **详细描述：**
  - 升级指定的模型PaddleNLP模型结构，每个模型的主要工作为：
    - 为模型结构增加configuration.py, 对齐huggingface/transformers的config，并且适配在模型代码中适配config, 详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
    - 为模型增加单测, 并且做到单测通过，详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
  - 每个模型算单独的子任务，**每升级3个模型算完成一个基础任务**, 总共待升级的模型为
    - [prophetnet](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/prophetnet)
    - [artist](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/artist)
    - [luke](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/luke)
  - 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)

- **提交内容：**
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法

### No.101：升级paddlenlp.transformers内的模型结构并且增加基础单测 <a name='task101'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：**基础⭐️
- **详细描述：**
  - 升级指定的模型PaddleNLP模型结构，每个模型的主要工作为：
    - 为模型结构增加configuration.py, 对齐huggingface/transformers的config，并且适配在模型代码中适配config, 详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
    - 为模型增加单测, 并且做到单测通过，详情见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4575)
  - 每个模型算单独的子任务，**每升级3个模型算完成一个基础任务**, 总共待升级的模型为
    - [blenderbot](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/blenderbot)
    - [blenderbot_small](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/blenderbot_small)
    - [ernie_doc](https://github.com/PaddlePaddle/PaddleNLP/blame/develop/paddlenlp/transformers/ernie_doc)
  - 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)

- **提交内容：**
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法

### No.102：给AutoConverter增加新的模型组网的支持 <a name='task102'></a>

- **任务难度：**基础⭐️
- **详细描述：**

  - 为PaddleNLP的AutoConverter增加支持的模型结构，使得更多的PaddleNLP模型可以无缝一行代码加载HuggingFace Hub上的torch模型
  - 每个模型算单独的子任务，总共待升级的模型共5个，完成全部5个模型算完成一个基础任务
    - [clip](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/clip)
    - [distilbert](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/distilbert)
    - [bart](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/bart)
    - [albert](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/albert)
    - [electra](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/electra)

  - 每个模型添加CompatibilityTest, 能够完成hf-internal-testing内相应torch模型的自动转换与精度对齐，见[范例PR](https://github.com/PaddlePaddle/PaddleNLP/pull/4477)
  - 开发流程和环境配置请参考 [CONTRIBUTING.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/CONTRIBUTING.md)，开发资源可以考虑使用[GitHub Codespaces](https://github.com/codespaces)

- **提交内容：**
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法

### No.103：新增tie_weights能力 <a name='task103'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：基础⭐️**
- **详细描述：**
  - 为PaddleNLP新增tie_weights功能，能够对齐HuggingFace Transformers中的[tie_weights](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.tie_weights)功能
- **提交内容：**
  - RFC文档：参考[API开发RFC模板](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/rfcs/api_design_template.md)，提交到 [目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/docs/community/rfcs)
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法


### No.104：生成式API对齐HF，包括sample和contrastive_search <a name='task104'></a>

- **技术标签：深度学习、Python、NLP**
- **任务难度：基础⭐️⭐️**
- **详细描述：**
  - PaddleNLP[生成式API](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.5.0/paddlenlp/transformers/generation_utils.py)功能对齐HuggingFace Transformers，重构[sample](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/generation/utils.py#L2259)，新增[contrastive_search](https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/generation/utils.py#L1659)
- **提交内容：**
  - RFC文档：参考[API开发RFC模板](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/community/rfcs/api_design_template.md)，提交到 [目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/docs/community/rfcs)
  - Python 实现代码，通过所有PaddleNLP的单元测试和代码风格测试，合入PaddleNLP
  - 参考
    - https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/generation/utils.py#L2259
    - https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/generation/utils.py#L1659
- **技术要求：**
  - 熟练掌握Python开发
  - 熟悉 HuggingFace, PaddleNLP或者相关NLP模型算法

### No.105：基于PaddleNLP PPDiffusers 训练 AIGC 趣味模型 <a name='task105'></a>

- **技术标签：**Python、NLP、扩散模型
- **奖励设置：**
  - 创意奖：一等奖奖金3k（1名），二等奖奖金1k（5名）；三等奖 面值 200 元京东卡（10名）
  - 参与奖：证书荣誉、飞桨周边礼品、 HF store 代金券、 HF pro 账号等
- **详细描述：**结合 [PPDiffusers](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers) 最新版本，基于自己的数据集，训练并开源趣味模型。可参考[模型训练 QuickStart](https://aistudio.baidu.com/aistudio/projectdetail/5513258)
  - 基于 [DreamBooth + LoRA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreambooth) 使用自己的图片 Fine-tune 模型；当然你也可以使用 PPDiffusers 中 Textual Inversion 等更多玩法，等你探索！
  - 主题推荐：
    - 流量地球
    - 三体
    - 表情包
    - 家乡景观
    - 动物萌宠
    - 家居设计
    - 无限想象（主题不限）
- **提交内容：**
  - 【必需】提交1个AI studio项目，遵从[模板](https://aistudio.baidu.com/aistudio/projectdetail/5519383)规范
  - 【必需】将模型文件上传到Hugging Face，在模型卡片中介绍模型，并上传由模型生成的图片（以及相应Prompt，至少3组）
  - 【可选】在Hugging Face Space搭建应用中心、跑通Inference API
- **提交流程：**
  - 在 [该Issue](https://github.com/PaddlePaddle/PaddleNLP/issues/4775) 下按如下模板回复，提交自己的趣味创意

```plain
【队名】：一个让人印象深刻的名字
【模型简介】：一句话描述自己模型的特色
【模型链接】：Hugging Face 地址
【AI Studio 项目地址】：xxx 
【可选】【Hugging Face 应用中心】：xxx
```

- **评奖：**
  - AI Studio 项目like数 + Fork数/100 + Hugging Face like数，结合AI Studio项目质量（实现思路、创意、项目作为教程的易读易用性）综合评选
  - 每支队伍可以提交多个模型，选择最好成绩作为队伍最终成绩
  - 只要提交，即可获得参与奖；每支队伍仅可获得一项创意奖
- **技术要求：**
  - 会用Python

### No.106：论文名称：Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion <a name='task106'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 概述：infer阶段增强SD等模型生成效果，解决catastrophic neglect和incorrect attribute inbinding问题
  - 完成算法复现对齐
  - 参考代码：https://github.com/AttendAndExcite/Attend-and-Excite
- **验收标准：**
  - 基于PPDiffusers复现论文和repo中的方法
  - 产出论文/repo中的case和Quantitative Analysis中的定量结果
  - 提交PR至PaddleNLP并且合入
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉扩散模型相关算法；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现。


### No.107：论文名称：Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation <a name='task107'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 概述：Text-Driven Image-to-Image Translation
  - 完成算法复现对齐
  - 参考代码：https://github.com/MichalGeyer/plug-and-play
- **验收标准：**
  - 基于PPDiffusers复现论文和repo中的方法
  - 产出论文/repo中的case和Quantitative Analysis中的定量结果
  - 提交PR至PaddleNLP并且合入
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉扩散模型相关算法；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现

### No.108：论文名称：AudioLDM: Text-to-Audio Generation with Latent Diffusion Models <a name='task108'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 概述：Generate speech, sound effects, music and beyond.
  - 完成算法复现对齐
  - 参考代码：https://github.com/haoheliu/AudioLDM
- **验收标准：**
  - 基于PPDiffusers复现论文和repo中的方法
  - 产出论文/repo中的case和Quantitative Analysis中的定量结果
  - 提交PR至PaddleNLP并且合入
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉扩散模型相关算法；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现


### No.109：论文名称：Zero-shot Image-to-Image Translation <a name='task109'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 概述：a diffusion-based image-to-image approach that allows users to specify the edit direction on-the-fly
  - 完成算法复现对齐
  - 参考代码：https://github.com/pix2pixzero/pix2pix-zero 
- **验收标准：**
  - 基于PPDiffusers复现论文和repo中的方法
  - 产出论文/repo中的case和Quantitative Analysis中的定量结果
  - 提交PR至PaddleNLP并且合入
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉扩散模型相关算法；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现。


### No.110：论文名称：Multi-Concept Customization of Text-to-Image Diffusion <a name='task110'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 概述：类似DreamBooth的扩散模型定制化微调，支持Multi-Concept且轻量化微调
  - 完成算法复现对齐
  - 参考代码：https://www.cs.cmu.edu/~custom-diffusion/
- **验收标准：**
  - 基于PPDiffusers复现论文和repo中的方法
  - 产出论文/repo中的case和Quantitative Analysis中的定量结果
  - 提交PR至PaddleNLP并且合入
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉扩散模型相关算法；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现


### No.111：论文名称：DeBERTa: Decoding-enhanced BERT with Disentangled Attention <a name='task111'></a>

- **技术标签：Python、深度学习、NLP**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 数据集： GLUE SQuAD v1.1
  - 完成算法复现对齐
- **验收标准：**
  - 复现DeBERTa和DeBERTa-v2（参考论文和实现链接）, 加入paddlenlp/transformers模块内
  - 适配AutoConverter, 以下模型能够实现权重从torch到paddle的自动转换
    - hf-internal-testing/tiny-random-DebertaV2Model (用于单测)
    - hf-internal-testing/tiny-random-DebertaModel (用于单测)
    - microsoft/deberta-base
    - microsoft/deberta-base-mnli
    - microsoft/deberta-v2-xlarge
    - microsoft/deberta-v3-base
  - 在下游任务上验证模型指标，加入examples/language_model/deberta内
    - microsoft/deberta-large: GLUE测试集上MNLI-m/mm=91.1/91.1（见论文Table 1），SQuAD v1.1验证集上F1/EM=95.5/90.1（见论文table 2）
    - microsoft/deberta-v2-xlarge: GLUE测试集上MNLI-m/mm=91.7/91.6, SQuAD v1.1验证集上F1/EM=96.1/91.4（见https://huggingface.co/microsoft/deberta-v2-xlarge）
- **提交内容：**
  - 提交PR包括：代码、模型、训练日志、中英文文档
- **技术要求：**
  - 熟悉NLP模型；有基本的模型训练和debug能力
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在飞桨导师的指导下完成复现

### No.112：论文复现：Multi-Granularity Prediction for Scene Text Recognition <a name='task112'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880336.pdf
  - 参考repo：https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR
  - 验收标准：基于MJ和ST数据集训练，在IC13 SVT IIIT IC15 SVTP CUTE上评估平均精度达到93.35%（论文中Table 6）。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.113：论文复现：PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition <a name='task113'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2207.14807
  - 参考repo：https://github.com/shannanyinxiang/PageNet
  - 验收标准：在ICDAR13上，指标对齐论文中的Table 2。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.114：论文复现：GLASS: Global to Local Attention for Scene-Text Spotting <a name='task114'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2208.03364
  - 参考repo：https://github.com/amazon-research/glass-text-spotting
  - 验收标准：
    - ToTalText上：E2E Hmean=79.9%，FPS=10
    - ICDAR2015：E2E Hmean=84.7%
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.115：论文复现：TPSNet: Reverse Thinking of Thin Plate Splines for Arbitrary Shape Scene Text Representation <a name='task115'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/pdf/2110.12826.pdf
  - 参考repo：https://github.com/Wei-ucas/TPSNet
  - 验收标准：
    - ICDAR2015上：hmean=89.1，fps=11.6
    - CTW1500上：hmean=87.5，fps=17.9
    - Total-Text上：hmean=88.5，fps=14.3
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.116：论文复现：ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting <a name='task116'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2105.03620
  - 参考repo：https://github.com/aim-uofa/AdelaiDet/
  - 验收标准：
    - ToTalText上：E2E Hmean=70.4%，FPS=10
    - SCUT-CTW1500：E2E Hmean=57.5%
    - ICDAR2015：E2E Hmean=:82.7
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.117：论文复现：CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition <a name='task117'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2207.04410
  - 参考repo：https://github.com/Green-Wood/CoMER
  - 验收标准：CROHME数据集上，指标对齐论文中的Table 2。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.118：论文复现：Syntax-Aware Network for Handwritten Mathematical Expression Recognition <a name='task118'></a>

- **技术标签：Python、深度学习**

- **任务难度：基础**️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2203.01601
  - 参考repo：https://github.com/tal-tech/SAN
  - 验收标准：CROHME数据集上，指标对齐论文中的Table 2。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.119：论文复现：Learning From Documents in the Wild to Improve Document Unwarping <a name='task119'></a>

- **技术标签：Python、深度学习**

- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 论文：https://drive.google.com/file/d/1z_8YaCc3aGWTHqFP55vgpSaEBz_oaQQe/view
  - 参考repo：https://github.com/cvlab-stonybrook/PaperEdge
  - 验收标准：DocUNet数据集上，指标对齐论文中的Table 2。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.120：论文复现：C3-STISR: Scene Text Image Super-resolution with Triple Clues <a name='task120'></a>

- **技术标签：Python、深度学习**

- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 论文：https://arxiv.org/abs/2204.14044
  - 参考repo：https://github.com/zhaominyiz/C3-STISR
  - 验收标准：TextZoom数据集上，指标对齐论文中的Table 1。
- **提交内容：**
  - 代码、模型、训练日志
  - 提交代码和中英文文档PR到PaddleOCR，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md)
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和debug能力。
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.121：PaddleOCR js部署 <a name='task121'></a>

- **技术标签：JavaScript、C++**

- **任务难度：进阶**⭐️⭐️
- **详细描述：通过c++打包PP-OCR多语言模型与PP-Structure版面恢复功能，完成js调用示例，开发js demo展示上述能力，撰写使用文档**
- **提交内容：**
  - PP-OCR多语言与PP-Structure c++打包模型
  - js demo代码：展示OCR可视化结果、语种选择、版面恢复等功能
  - 中英文文档说明，在PaddleOCR repo的deploy/paddlejs目录
- **技术要求：**
  - 熟练掌握JavaScript和相关框架，有通过js应用AI能力者优先
  - 掌握c++打包过程，能够根据已有文档独立完成模型打包

### No.122：《动手学OCR》升级 <a name='task122'></a>

- **技术标签：Python、文档、OCR**
- **任务难度：基础**️⭐️
- **详细描述：结合PaddleOCR最新版本，更新****《动手学OCR》中的相关代码，验证notebook可跑通，整合已有资料新增PP-OCRv3、PP-StructureV2章节内容**
- **提交内容：**
  - 更新章节的notebook，再Dive into OCR repo的相关目录
- **技术要求：**
  - 熟练掌握Python，撰写过notebook项目
  - 掌握PP-OCRv3、PP-StructureV2的功能与实现流程

### No.123：模型库中文适配 <a name='task123'></a>

- **技术标签：Python、深度学习、OCR**
- **任务难度：基础**️⭐️
- **详细描述：从PaddleOCR中的模型库（文本检测、文本识别、端到端）中挑选2种算法适配中文场景（数据集ICDAR2019-LSVT），得到benchmark。**
- **提交内容：**
  - 模型、配置文件、训练日志
  - 文档：含训练过程描述（主要做了哪些适配），benchmark指标
- **技术要求：**
  - 熟悉OCR领域相关算法，最好有PaddleOCR使用经验。
  - 有基本的模型训练和调参能力。
  - 参加中文适配的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.124：论文复现：More ConvNets in the 2020s: Scaling up Kernels Beyond 51x51 using Sparsity <a name='task124'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型SLaK-T(51x51 kernel size)精度达到82.5%；
  - 补全系列模型SLaK-T、SLaK-S、SLaK-B配置和预训练模型文件；
  - 参考repo：https://github.com/VITA-Group/SLaK
  - 论文：https://arxiv.org/pdf/2207.03620.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.125：论文复现：Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs <a name='task125'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型RepLKNet-31B精度达到83.58%；
  - 补全模型RepLKNet-31B配置和预训练模型文件；
  - 参考repo：https://github.com/MegEngine/RepLKNet
  - 论文：https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.126：论文复现：Revisiting ResNets: Improved Training and Scaling Strategies <a name='task126'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型ResNet-RS-50-i160精度达到79.1%；
  - 补全系列模型ResNet-RS-500-i160、ResNet-RS-101-i160、ResNet-RS-101-i192、ResNet-RS-152-i192、ResNet-RS-152-i224、ResNet-RS-152-i256、ResNet-RS-200-i256、ResNet-RS-270-i256、ResNet-RS-350-i256、ResNet-RS-350-i320、ResNet-RS-420-i320配置和预训练模型文件；
  - 参考repo：https://github.com/tensorflow/models/tree/master/
  - 论文：https://proceedings.neurips.cc/paper/2021/file/bef4d169d8bddd17d68303877a3ea945-Paper.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.127：论文复现：Separable Self-attention for Mobile Vision Transformers <a name='task127'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型MobileViTv2-1.5精度达到80.4%；
  - 补全系列模型MobileViTv2-0.5、MobileViTv2-1.0、MobileViTv2-1.5、MobileViTv2-2.0配置和预训练模型文件；
  - 参考repo：
    https://github.com/apple/ml-cvnets
  - 论文：https://arxiv.org/pdf/2206.02680.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。


### No.128：论文复现：MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features <a name='task128'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型MobileViTv3-S精度达到79.3%；
  - 补全系列模型MobileViTv3-S、MobileViTv3-XS、MobileViTv3-XXS、MobileViTv3-1.0、MobileViTv3-0.75、MobileViTv3-0.5、MobileViTv3-S-L2、MobileViTv3-XS-L2、MobileViTv3-XXS-L2配置和预训练模型文件；
  - 参考repo：https://github.com/micronDLA/MobileViTv3
  - 论文：https://github.com/micronDLA/MobileViTv3
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.129：论文复现：Model Rubik’s Cube: Twisting Resolution, Depth andWidth for TinyNets <a name='task129'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型TinyNet-A精度达到76.8%；
  - 补全系列模型TinyNet-A、TinyNet-B、TinyNet-C、TinyNet-D、TinyNet-E配置和预训练模型文件；
  - 参考repo：https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tinynet_pytorch
  - 论文：https://arxiv.org/pdf/2010.14819.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.130：论文复现：FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling <a name='task130'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于CIFAR-100(4 label samples per class)数据集，模型FlexMatch精度达到60.06%；
  - 补全模型配置和预训练模型文件；
  - 参考repo：https://github.com/TorchSSL/TorchSSL
  - 论文：https://arxiv.org/abs/2110.08263
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.131：论文复现：Hypergraph-Induced Semantic Tuplet Loss for Deep Metric Learning <a name='task131'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于SOP数据集，模型HIST-ResNet-50 Recall@1达到89.6；
  - 补全系列模型配置和预训练模型文件；
  - 参考repo：https://github.com/ljin0429/HIST
  - 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Lim_Hypergraph-Induced_Semantic_Tuplet_Loss_for_Deep_Metric_Learning_CVPR_2022_paper.html
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.132：论文复现：iBOT: Image BERT Pre-Training with Online Tokenizer <a name='task132'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet1k数据集，模型iBOT-ViT-S/16(linear probing)精度达到77.9%；
  - 补全系列模型iBOT-ViT-S/16、iBOT-ViT-B/16、iBOT-ViT-L/16、iBOT-Swin-T/7、iBOT-Swin-T/14配置和预训练模型文件；
  - 参考代码：https://console.cloud.baidu-int.com/devops/icode/repos/baidu/personal-code/finger-train-code/blob/master/finger_train_code/ibot-run.tar.gz
  - 论文：http://export.arxiv.org/abs/2111.07832
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.133：论文复现：Forward Compatible Training for Large-Scale Embedding Retrieval Systems <a name='task133'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet-500/1k数据集，模型FCT-new-old CMC Top1达到65.0%；
  - 补全系列模型配置和预训练模型文件；
  - 参考repo：https://github.com/apple/ml-fct
  - 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Ramanujan_Forward_Compatible_Training_for_Large-Scale_Embedding_Retrieval_Systems_CVPR_2022_paper.html
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.134：论文复现：ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification <a name='task134'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于Market1501数据集，模型ICE(aware) mAP达到82.3；
  - 补全系列模型ICE(aware) 、ICE(agnostic)、ICE (w/ ground truth) 配置和预训练模型文件；
  - 参考repo：https://github.com/chenhao2345/ICE
  - 论文：https://arxiv.org/pdf/2103.16364.pdf
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.135：论文复现：VL-LTR: Learning Class-wise Visual-Linguistic Representation for Long-Tailed Visual Recognition <a name='task135'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于ImageNet-LT数据集和ResNet50模型，精度达到 70.1%；
  - 补全模型配置和预训练模型文件；
  - 参考repo：https://github.com/ChangyaoTian/VL-LTR
  - 论文：https://arxiv.org/abs/2111.13579
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.136：论文复现：Recall@k Surrogate Loss with Large Batches and Similarity Mixup <a name='task136'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 基于SOP数据集，精度达到 80.9%；
  - 补全模型配置和预训练模型文件；
  - 参考repo：https://github.com/yash0307/RecallatK_surrogate
  - 论文：https://arxiv.org/abs/2108.11179
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
  - PaddleClas合入规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.137：论文题目：Learning Transferable Visual Models From Natural Language Supervision(CLIP)论文题目：BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers <a name='task137'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 根据提供的项目代码，按照PaddleClas的规范整理、修改，最终将代码合入PaddleClas套件
  - 代码合入后，前反向完全对齐；在提供的小数据集上预训练并finetune后，精度对齐
  - 代码符合PaddleClas规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle，熟悉PaddleClas代码
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.138：论文题目：Expanding language-image pretrained models for general video recognition(X-CLIP)论文题目：Context Autoencoder for Self-Supervised Representation Learning(CAE) <a name='task138'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 根据提供的项目代码，按照PaddleClas的规范整理、修改，最终将代码合入PaddleClas套件
  - 代码合入后，前反向完全对齐；在提供的小数据集上预训练并finetune后，精度对齐
  - 代码符合PaddleClas规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle

### No.139：论文题目：BEIT: BERT Pre-Training of Image Transformers论文题目：Masked Autoencoders Are Scalable Vision Learners(MAE)论文题目：Exploring Simple Siamese Representation Learning(SimSam) <a name='task139'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 根据提供的项目代码，按照PaddleClas的规范整理、修改，最终将代码合入PaddleClas套件
  - 代码合入后，前反向完全对齐；在提供的小数据集上预训练并finetune后，精度对齐
  - 代码符合PaddleClas规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle，熟悉PaddleClas代码
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.140：论文题目：Improved baselines with momentum contrastive learning(MoCov2)论文题目：Bootstrap your own latent: A new approach to self-supervised learning（BYOL）论文题目：Unsupervised Learning of Visual Features by Contrasting Cluster Assignments （SwAV） <a name='task140'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 根据提供的项目代码，按照PaddleClas的规范整理、修改，最终将代码合入PaddleClas套件
  - 代码合入后，前反向完全对齐；在提供的小数据集上预训练并finetune后，精度对齐
  - 代码符合PaddleClas规范， [参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/add_new_algorithm.md)；
- **提交内容：**
  - 代码、训练模型文件、训练日志；
  - 代码提PR合入PaddleClas套件，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/community/how_to_contribute.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle，熟悉PaddleClas代码
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.141：PP-LCNet v3 下游场景验证 <a name='task141'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 使用PP-LCNet v3 完成不同的下游任务的验证，相比其他骨干网络，带来稳定收益
  - 下游任务具体包括：
    - 检测算法PP-PicoDet，[参考](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/picodet)；
    - 人脸识别算法，[参考](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_paddle)；
    - 商品识别算法，[参考配置](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml)，[参考文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/zh_CN/application/product_recognition.md)；
    - PP-Structure，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/README_ch.md)；
    - 属性识别（行人、车辆、人脸）至少一种，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/model_list.md)；
    - 二分类（有人/无人、有车/无车）至少一种，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/model_list.md)；
    - 多分类（交通标志、图像方向分类）至少一种，[参考](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/model_list.md)；
- **提交内容：**
  - 训练代码、模型、训练日志、和竞品的精度速度信息
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、图像分类、目标检测等算法

### No.142：PP-HGNet v2下游场景验证 <a name='task142'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 使用PP-HGNet v2完成不同的下游任务的验证，相比其他骨干网络，带来稳定收益
- **提交内容：**
  - 训练代码、模型、训练日志、和竞品的精度速度信息
  - 下游任务具体包括：
    - 语义分割PP-LiteSeg算法，[参考](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/configs/pp_liteseg)；
    - PP-OCR检测算法，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/ppocr_introduction.md#pp-ocrv3)；
    - PP-OCR识别算法，[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/ppocr_introduction.md#pp-ocrv3)；
    - PP-YOLO-E，[参考](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README_cn.md)；
    - PP-TSM，[参考](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)；
    - 人脸识别算法，[参考](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_paddle)；
    - ReID，[参考配置](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml)；
    - 关键点检测，[参考](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/advanced_tutorials/customization/keypoint_detection.md)；
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、图像分类、目标检测等算法

### No.143：万类通用识别数据集制作 <a name='task143'></a>

- **技术标签：Python、数据集、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：梳理开源检测、识别数据集中的种类标签，通过可视化等手段分析数据分布，统一标签格式，构造万级类目的通用识别数据集**
- **提交内容：**
  - 整理后的数据集与类目表
- **技术要求：**
  - 熟练掌握Python、了解多种常见的计算机视觉方向数据集

### No.144：SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation <a name='task144'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 在ADE20k分割数据集上，SeaFormer-Base 精度为40.2（参考论文Table.1）；并补全tiny，small，large配置和模型后合入PaddleSeg。
  - 参考repo https://github.com/fudan-zvg/SeaFormer
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleSeg套件
  - seg合入规范 [链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和语义分割
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.145：EfficientFormerV2：Rethinking Vision Transformers for MobileNet Size and Speed <a name='task145'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 在ADE20k分割数据集上，EfficientFormerV2-S2 精度为42.4（参考论文Table.3）；并补全S0，S1，L的配置和模型后合入PaddleSeg
  - 参考repo https://github.com/snap-research/EfficientFormer
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleSeg套件
  - seg合入规范 [链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和语义分割
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.146：Fully Convolutional Networks for Panoptic Segmentation <a name='task146'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在 Cityscapes 全景分割数据集的验证集上，复现得到 Panoptic FCN（R50-FPN）的 PQ 不低于 59.6（参考论文Table 13）
  - 为模型接入 TIPC 基础训推链条
  - 补全 COCO 数据集上的配置、精度（在验证集上评测）和预训练权重
  - 参考repo[：](https://github.com/fudan-zvg/SeaFormer)https://github.com/dvlab-research/PanopticFCN
- **提交内容：**
  - 代码、配置文件、预训练权重、训练日志；合入PaddleSeg 的 contrib/PanopticSeg 目录（遵循 PaddleSeg 套件代码合入规范：[链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)）
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、PaddleSeg，熟悉全景分割任务
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.147：Per-Pixel Classification is Not All You Need for Semantic Segmentation <a name='task147'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在 ADE20K 全景分割数据集的验证集上，复现得到 MaskFormer（R50 + 6 Enc）的 PQ 不低于 34.7（参考论文Table VI）
  - 参考repo：https://github.com/facebookresearch/MaskFormer
  - 参考 Paddle 实现的用于语义分割的 MaskFormer：https://github.com/PaddlePaddle/PaddleSeg/pull/2789
- **提交内容：**
  - 代码、配置文件、预训练权重、训练日志；合入PaddleSeg 的 contrib/PanopticSeg 目录（遵循 PaddleSeg 套件代码合入规范：[链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)）
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、PaddleSeg，熟悉全景分割任务
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.148：K-Net: Towards Unified Image Segmentation <a name='task148'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在 COCO 全景分割数据集的验证集上，复现得到 K-Net（R50-FPN）的 PQ 不低于 47.1（参考论文Table 1）
  - 参考repo：https://github.com/ZwwWayne/K-Net
- **提交内容：**
  - 代码、配置文件、预训练权重、训练日志；
  - 合入PaddleSeg 的 contrib/PanopticSeg 目录（遵循 PaddleSeg 套件代码合入规范：[链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)）
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、PaddleSeg，熟悉全景分割任务
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.149：Highly Accurate Dichotomous Image Segmentation （ECCV 2022） <a name='task149'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在论文提出的DIS5K数据集上，复现IS-Net模型和HCE评估指标，六个评估指标达到论文Table.2中所示精度
  - 参考repo https://github.com/xuebinqin/DIS
  - 论文 https://arxiv.org/abs/2203.03041
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleSeg套件
  - seg合入规范 [链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/pr/pr/style_cn.md)
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、PaddleSeg和语义分割
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.150：PaddleRS集成PaddleDetection的旋转框检测能力 <a name='task150'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - PaddleRS是针对遥感影像智能解译的开发套件，已集成有语义分割、水平框目标检测等能力，现需集成PaddleDetection套件已有的旋转框目标检测能力，并提供说明文档
  - 参考：https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rotate/README.md
- **提交内容：**
  - 代码、文档，合入PaddleRS套件
  - PaddleRS贡献指南 [链接](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/CONTRIBUTING.md)
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、PaddleRS和目标检测

### No.151：PaddleRS运行环境打包，并制作端到端遥感建筑物提取教程 <a name='task151'></a>

- **技术标签：Python、docker、文档**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 打包制作包含PaddleRS、EISeg、GeoView最新版本运行环境的docker镜像
  - 利用官方提供的示例遥感影像，基于前述镜像环境，跑通从数据标注（EISeg）、模型训练（PaddleRS）、部署应用（GeoView）的遥感建筑物提取全流程，并形成教程文档。
- **提交内容：**
  - docker镜像、教程文档
- **技术要求：**
  - 熟练掌握Python、docker、PaddleRS，熟悉遥感场景

### No.152：PaddleRS API 文档完善 <a name='task152'></a>

- **技术标签：文档**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 撰写中英文文档，对 PaddleRS 每个模型的构造参数进行详细说明（参考 paddlers/rs_models 中的 docstring）
  - 撰写中英文文档，对 PaddleRS 每个数据预处理/数据变换算子的构造参数进行详细说明
  - 完善 PaddleRS quick start 中英文文档，在已有的『模型训练』部分内容基础上，添加『模型精度验证』与『模型部署』部分内容。
- **提交内容：**
  - 教程文档
- **技术要求：**
  - 熟悉 PaddleRS

### No.153：PaddleRS 英文文档 <a name='task153'></a>

- **技术标签：文档**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 将 PaddleRS docs 目录中包含的文档内容翻译为英文，并形成相应英文文档
  - 为 PaddleRS tutorials 目录中的示例脚本添加英文注释
- **提交内容：**
  - 教程文档
- **技术要求：**
  - 熟悉 PaddleRS，外语水平良好

### No.154：论文复现：YOLOv6 v3.0: A Full-Scale Reloading <a name='task154'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐️
- **详细描述：**
  - s版本COCO数据集精度37.5mAP
  - 完成复现后合入PaddleYOLO
  - 参考repo https://github.com/meituan/YOLOv6
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleYOLO套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.155：YOLOv8模型复现 <a name='task155'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐️
- **详细描述：**
  - YOLOv8-n版本COCO数据集精度37.2mAP
  - 完成复现后合入PaddleYOLO
  - 参考repo https://github.com/ultralytics/ultralytics
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleYOLO套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.156：论文复现：Open-Vocabulary DETR with Conditional Matching <a name='task156'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在 Open-vocabulary COCO (AP50 metric) base类上AP50指标达到61.0（原github仓库权重已开源）
  - 完成复现后合入PaddleDetection
  - 参考repo https://github.com/yuhangzang/OV-DETR
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.157：论文复现：PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection <a name='task157'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 10 % unlabelled coco data，精度达到36.06%
  - 完成复现后合入PaddleDetection
  - 参考repo https://github.com/ligang-cs/PseCo
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.158：论文复现：DiffusionDet: Diffusion Model for Object Detection <a name='task158'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - COCO数据集 Box AP 45.5
  - 完成复现后合入PaddleDetection
  - 参考repo https://github.com/ShoufaChen/DiffusionDet
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.159：论文复现：PromptDet: Towards Open-vocabulary Detection using Uncurated Images <a name='task159'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - LVIS 数据集 AP 21.3，APnovel 19.5
  - 完成复现后合入PaddleDetection
  - 参考repo https://github.com/fcjian/PromptDet
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.160：论文复现：Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone <a name='task160'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 只复现在coco上finetune阶段 COCO Box AP 58.4
  - 完成复现后合入PaddleDetection
  - 参考repo https://github.com/microsoft/FIBER
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.161：论文复现：CLRNet: Cross Layer Refinement Network for Lane Detection <a name='task161'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐️
- **详细描述：**
  - ps2.0数据集，precision 99.56%
  - 参考repo https://github.com/Jiaolong/gcn-parking-slot
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.162：论文复现：Attentional Graph Neural Network for Parking Slot Detection <a name='task162'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**️⭐️
- **详细描述：**
  - CULane数据集，backbone RestNet18版本，mF1指标达到55.23
  - 参考repo https://github.com/Turoad/CLRNet
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.163：基于PaddleDetection PP-TinyPose，新增手势关键点检测模型 <a name='task163'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐️
- **详细描述：**
  - 基于PaddleDetection的PP-TinyPose算法，新增手势关键点检测模型，适配COCO-Whole-Body hand数据集
  - 参考文档：
    - PP-TinyPose： https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose
    - COCO-Whole-Body hand：
      - https://github.com/jin-s13/COCO-WholeBody/
      - https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py
- **提交内容：**
  - 代码、模型、训练日志，合入PaddleDetection套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和目标检测算法

### No.164：PaddleDetection重点模型接入huggingface <a name='task164'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐️
- **详细描述：**
  - PP-YOLOE+，PP-YOLOE，PP-PicoDet，PP-Human，PP-Vehicle接入huggingface中，并在PaddleDetection中提供使用文档
  - 可以参考https://huggingface.co/spaces/PaddlePaddle/PaddleOCR/tree/main 和 https://github.com/PaddlePaddle/models/tree/release/2.3/modelcenter/PP-YOLOE%2B
- **提交内容：**
  - huggingface模型页面
  - PaddleDetection使用文档
- **技术要求：**
  - 熟练掌握PaddleDetection目标检测算法，以及huggingface的开发逻辑
  - 熟悉gradio

### No.165：Camera标定LiDAR标定 <a name='task165'></a>

- **技术标签：Python、C++、ROS、OpenCV**
- **任务难度：基础**⭐️
- **详细描述：**
  - 开发相机、激光雷达内外参标定算法
  - 提供标定模版文件
  - 提供标定流程说明以及运行demo
  - 标定算法稳定且标定流程简单
- **提交内容：**
  - 算法代码、标定流程文档、标定模版，标定演示视频，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握相机、激光雷达标定流程、拥有相机、激光雷达标定实操经验

### No.166：Paddle3D目标检测结果可视化 <a name='task166'></a>

- **技术标签：Python**
- **任务难度：基础**⭐️
- **详细描述：**
  - 将单目，BEV，点云检测结果可视化
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/open-mmlab/mmdetection3d/tree/master/demo
- **提交内容：**
  - 代码、文档，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和3D目标检测算法

### No.167：Paddle3D&ROS联合开发Demo <a name='task167'></a>

- **技术标签：Python、C++、ROS、OpenCV**
- **任务难度：基础**⭐️
- **详细描述：**
  - 将Paddle3D中的已有算法适配ROS，并提供rviz可视化调试能力
  - 编写ROS安装、运行文档
  - 编写ROS联合开发文档
  - 提供Paddle3D&ROS联合开发算法示例(C++和Python)
- **提交内容：**
  - 算法代码、开发文档、运行展示
- **技术要求：**
  - 熟练掌握相机标定流程、拥有相机标定实操经验

### No.168：Geometry Uncertainty Projection Network for Monocular 3D Object Detection <a name='task168'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - KITTI验证集AP40CarEasy23.19%Mod16.23%Hard13.57%
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/SuperMHP/GUPNet
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和3D目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.169：DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries <a name='task169'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - nuscenes 验证集NDS指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/WangYueFt/detr3d
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和3D, BEV目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.170：TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers <a name='task170'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - nuscenes 验证集NDS指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/XuyangBai/TransFusion/
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和3D, BEV，点云目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.171：FUTR3D: A Unified Sensor Fusion Framework for 3D Detection <a name='task171'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - nuscenes 验证集NDS指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/Tsinghua-MARS-Lab/futr3d
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle和3D, BEV，点云目标检测算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.172：Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction <a name='task172'></a>

- **技术标签：Python、深度学习、CUDA编程**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - NeRF Synthetic验证集PSNR指标
  - 以Paddle自定义C++算子的形式复现repo中的cuda算子（部分算子已经开发完成）
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/sunset1995/DirectVoxGO
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle（包括自定义算子开发）、神经渲染算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.173：Point-based Neural Radiance Fields <a name='task173'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - NeRF Synthetic验证集PSNR指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/Xharlie/pointnerf
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、神经渲染算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.174：Scenes as Neural Radiance Fields for View Synthesis <a name='task174'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - NeRF Synthetic验证集PSNR指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/bmild/nerf
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、神经渲染算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.175：TensoRF: Tensorial Radiance Fields <a name='task175'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - NeRF Synthetic验证集PSNR指标
  - 完成复现后合入Paddle3D
  - 参考repo https://github.com/apchenstu/TensoRF
- **提交内容：**
  - 代码、模型、训练日志，合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Python、PaddlePaddle、神经渲染算法
  - 参加模型复现的同学需先发送简历和想复现的文章（可多选）到paddle-lwfx <paddle-lwfx@baidu.com>报名，通过筛选后锁定题目，在Paddle导师的指导下完成复现。

### No.176：相机去畸变C++自定义算子开发 <a name='task176'></a>

- **技术标签：Python、CUDA编程**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 开发Paddle C++自定义算子，替换Paddle3D已有的相机去畸变函数，做到精度一致、性能提升
  - 完成复现后合入Paddle3D
  - 参考代码：待提供
- **提交内容：**
  - 代码合入Paddle3D套件
- **技术要求：**
  - 熟练掌握Paddle自定义算子开发

### No.177：将PP-YOLOE-R在**算能BM1684**部署。利用FastDeploy，将PP-YOLOE-R在**算能BM1684X**部署 <a name='task177'></a>

- **技术标签：**深度学习，C++、Python
- **任务难度：**基础⭐️ 
- **详细描述：**
  - 需要完成PP-YOLOE-R，算法前后处理,开发Python部署示例和C++部署示例
  - 模型repo：[PaddleDetection/release/2.6/configs/rotate/ppyoloe_r](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r)
  - 模型repo python推理脚本：[configs/rotate/tools/onnx_infer.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/rotate/tools/onnx_infer.py)
  - 模型导出文档：[deploy/EXPORT_MODEL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_MODEL.md)
  - 算能详细参考链接：接 https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo
  - FD模型开发文档：[develop_a_new_model.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/develop_a_new_model.md)
  - 提示：特别需要注意multiclass_nms_rotated和rbox_iou的处理细节。
- **提交内容：**
  - 提交Python和C++实现代码 ，在fastdeploy/vision/detection/ppdet中增加支持PP-YOLOE-R的代码逻辑，并在ppdet_pybind.cc中绑定C++模型到python，在python/fastdeploy/vision/detection/ppdet/__init__.py中增加python类；
  - 提交使用案例，在FastDeploy repo 的[examples/vision/detection/paddledetection/sophgo](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)；
  - 中英文文档，在FastDeploy repo 的[examples/vision/detection/paddledetection/sophgo](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)；包含从零的环境安装文档。
  - 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。
  - 将模型提交到算能model zoo
- **技术要求：**
  - 熟练掌握Python和C++开发，了解AI部署
  - 了解算能BM1684
  - 了解旋转目标检测算法

### No.178：集成SOLOv2模型到FastDpeloy，并在Paddle Infenence、ONNX Runtime、TernsorRT后端测试验证 <a name='task178'></a>

- **技术标签：**深度学习，C++、Python
- **任务难度：**基础⭐️
- **详细描述：**
  - 完成SOLOv2算法前后处理，及精度对齐，开发Python部署示例和C++部署示例
  - 模型repo：[PaddleDetection/tree/release/2.6/configs/solov2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/solov2)
  - 模型repo 后处理：[deploy/cpp/src/object_detector.cc](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/cpp/src/object_detector.cc#L250)
  - 模型导出文档：[deploy/EXPORT_MODEL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/EXPORT_MODEL.md)
  - FD模型开发文档：[develop_a_new_model.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/develop_a_new_model.md)
  - 提示：特别需要注意mask的处理细节。
  - 需要开发Python部署示例和C++部署示例，详细参考链接[https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)python、C++
  - 进阶要求（非必需）：将FastDeploy联合ros编译，完成机器人的自动避障。
- **提交内容：**
  - Python和C++实现代码 ，在FastDeploy repo 的[examples/vision/detection/paddledetection/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)jetson；
  - 中英文文档，在FastDeploy repo 的[examples/vision/detection/paddledetection/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)jetson；包含从零的环境安装文档
  - 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。
- **技术要求：部署**
  - 熟练掌握C++、Python开发，了解AI
  - 了解jetson orin
  - 了解实例分割算法

### No.179：将[PointPillars](https://github.com/PaddlePaddle/Paddle3D/blob/release/1.0/docs/models/pointpillars)集成到FastDeploy，并在**Jetson Orin**硬件上部署验证精度和速度 <a name='task179'></a>

- **技术标签：**深度学习，C++、Python
- **任务难度：**进阶⭐️⭐️
- **详细描述：**
  - 完成SOLOv2算法前后处理，及精度对齐，开发Python部署示例和C++部署示例
  - 模型repo：https://github.com/PaddlePaddle/Paddle3D
  - FD模型开发文档：[develop_a_new_model.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/develop_a_new_model.md)
  - 提示：特别需要注意mask的处理细节。
  - 需要开发Python部署示例和C++部署示例，详细参考链接[https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddle3d/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)python、C++
  - 进阶要求（非必需）：将FastDeploy联合ros编译，完成机器人的自动避障。
- **提交内容：**
  - Python和C++实现代码 ，在FastDeploy repo 的[examples/vision/detection/paddle3d/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)jetson；
  - 中英文文档，在FastDeploy repo 的[examples/vision/detection/paddle3d/](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection/sophgo)jetson；包含从零的环境安装文档
  - 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。
- **技术要求：部署**
  - 熟练掌握C++、Python开发，了解AI
  - 了解jetson orin
  - 了解3D算法

### No.180：在FastDeploy中集成集成**地平线**推理引擎，在PP-YOLOE完成模型转换测试 <a name='task180'></a>

- **技术标签：**深度学习，C++
- 任务难度：进阶⭐️⭐️
- 详细描述:
  - 需要完成地平线AI推理引擎接入Fastdeploy工作，并转换PP-YOLOE后，测试模型运行正确
  - 推理后端接入：直接通过ONNX接入自己的推理引擎，支持Paddle/ONNX模型（TensorRT是此方案）；可以先看下fastdeploy/runtime/backends/tensorrt中接入的代码，有问题可随时沟通
- **提交内容：**
  - 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。
  - 推理后端
- **技术要求：**
  - 熟练掌握C++开发
  - 熟练使用地平线推理工具链完成AI模型部署。

### No.181：完成**TVM**接入FastDeploy，并在PP-YOLOE模型上验证正确性 <a name='task181'></a>

- **技术标签：**深度学习，C++
- **任务难度**：进阶⭐️⭐️
- **详细描述:**
  - 需要完成TVM接入FastDeploy工作，并完成PP-YOLOE基于TVM后端的测试，确定精度和推理速度正确
- **提交内容：**
  - pr：提交适配代码，及对应的中英文文档
  - pr：提交PP-YOLOE的部署示例
  - 邮件：提交benchmark测试数据及精度对齐数据。
- **技术要求：**
  - 熟练掌握C++开发。
  - 熟练使用TVM完成AI模型部署。

### No.182：完成pp-ocrv3在**RK3588**上的部署，并验证正确性 <a name='task182'></a>

- **技术标签：**深度学习，C++
- **任务难度**：进阶⭐️⭐️
- **详细描述**：
  - 完成PP-OCRv3模型转换，并量化完成在在RK3588上的部署，确定精度和推理速度正确
- **提交内容**：
  - pr：提交适配代码，及对应的中英文文档
  - pr：提交PP-YOLOE的部署示例
  - 提交benchmark测试数据及精度对齐数据。
- **技术要求**：
  - 熟练掌握C++开发。
  - 熟练使用RKNN、FastDeploy完成AI模型部署。
  - 熟悉PP-OCRv3多模型串联算法细节。

### No.183：使用FastDeploy完成 ELECTRA 模型GLUE任务模型部署 <a name='task183'></a>

- 技术标签：深度学习，C++，Python
- 任务难度：进阶⭐️
- 详细描述：
  - 基于FastDeploy在GLUE任务上完成 ELECTRA 模型的部署，包含C++、Python两种部署方式。
  - 参考内容
    - ERNIE 3.0 FastDeploy 部署： https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0/deploy
    - ERNIE Tiny FastDeploy 部署： https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-tiny/deploy
- **提交内容：**
  - PR：包含GLUE任务上ELECTRA模型的C++、Python部署代码及文档、测试脚本。
  - 验收标准：
    - 目录结构：参考ernie 3.0。deploy目录下分别包含README文档，cpp、python目录。各个目录下包含相应的README文档以及代码。
    - 代码：包含 Python、C++ 代码
    - 文档：描述清楚每一种部署方式如何运行，包括环境依赖、示例代码编译及运行，示例的参数等内容。
    - 测试：在脚本(scripts/regression/run_deploy.sh)新增测试内容，验证BERT模型Python 部署示例能否正常运行；C++部署示例能否正常编译、运行；PR会有 CI 验证运行该脚本，运行成功则通过。
- **技术要求：**
  - 熟练掌握 C++、Python 开发
  - 熟练使用FastDeploy等推理引擎完成模型部署
  - 熟悉 NLP 模型文本分类任务部署，包括文本预处理、模型后处理细节

### No.184：在FastDeploy C API的基础上，使用rust完成PaddleDetection部署 <a name='task184'></a>

- **技术标签：**深度学习，Rust
- 任务难度：**基础⭐️**
- 详细描述:
  - 在Rust层，通过调用FastDeploy C API，完成PP-YOLOE, PaddleYOLOv8, PaddleYOLOv5等模型的部署
- **提交内容：**
  - pr：提交适配代码，及对应的中英文文档，到FastDeploy repo下的examples/application/rust路径下。
- **技术要求：**
  - 熟练掌握Rust开发
  - 熟练AI模型部署。

### No.185：在FastDeploy C++ API的基础上，使用java完成PaddleDetection部署 <a name='task185'></a>

- **技术标签：**深度学习，java
- **任务难度**：**基础⭐️**
- **详细描述**:
  - 在Java层，通过JNI调用FastDeploy C++ API，完成PP-YOLOE, PaddleYOLOv8, PaddleYOLOv5等模型的部署
- **提交内容：**
  - pr：提交适配代码，及对应的中英文文档，到FastDeploy repo下的examples/application/java路径下。
- **技术要求：**
  - 熟练掌握java开发。
  - 熟练AI模型部署。

### No.186：在FastDeploy C API的基础上，使用go完成PaddleDetection部署 <a name='task186'></a>

- **技术标签：**深度学习，go
- **任务难度**：基础⭐️
- 详细描述:
  - 在go层，通过调用FastDeploy C API，完成PP-YOLOE, PaddleYOLOv8, PaddleYOLOv5等模型的部署
- **提交内容：**
  - pr：提交适配代码，及对应的中英文文档，到FastDeploy repo下的examples/application/go路径下。
- **技术要求：**
  - 熟练掌握go开发
  - 熟练AI模型部署。

### No.187：模型复现：pruned_transducer_stateless8 <a name='task187'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - **完成算法复现推理对齐，并按照PaddleSpeech新增算法规范提交PR。**
  - CER与预训练模型结果一致
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
  - 参考PR：https://github.com/PaddlePaddle/PaddleSpeech/pull/2640
- **技术要求：**
  - 熟悉语音识别 transformer 类模型相关算法。
  - 有基本的模型训练和debug能力。

### No.188：模型复现：hubert <a name='task188'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - **完成算法复现推理对齐，并按照PaddleSpeech新增算法规范提交PR。**
  - 在下游任务上结果一致
    - 下游任务任意：推荐 librispeech 下的语音识别任务
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
  - 参考PR：https://github.com/PaddlePaddle/PaddleSpeech/pull/2640
- **技术要求：**
  - 熟悉语音识别 transformer 类模型相关算法。
  - 有基本的模型训练和debug能力。

### No.189：模型复现：wavlm <a name='task189'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - **完成算法复现推理对齐，并按照PaddleSpeech新增算法规范提交PR。**
  - 在下游任务上结果一致。
    - 下游任务任意：推荐 librispeech 下的语音识别任务
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
  - 参考PR：https://github.com/PaddlePaddle/PaddleSpeech/pull/2640
- **技术要求：**
  - 熟悉语音识别 transformer 类模型相关算法。
  - 有基本的模型训练和debug能力。

### No.190：模型复现：iSTFTNet <a name='task190'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - **完成算法复现推理对齐，并按照PaddleSpeech新增算法规范提交PR。**
  - iSTFTNet 是 HiFiGAN 的改进，可以加快 HiFiGAN 的预测流程，同时可以应用于 VITS 模型中 ，请为 csmsc 数据集新增一个 iSTFTNet vocoder。
    - 请最大程度复用 PaddleSpeech HiFiGAN 的组网代码、数据预处理代码和训练代码，最好是直接修改 HiFiGAN 组网代码，通过 config 文件控制是使用 default HiFiGAN 还是 iSTFTNet，并且不影响 default HiFiGAN 的所有功能
    - 最好直接在 https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc5/conf 新增一个 yaml 文件用于控制 iSTFTNet，若实在难以实现可以在 https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc 新增 voc7
    - iSTFTNet 效果符合论文描述，即合成效果不差于 HiFiGAN、rtf < HiFiGAN
    - 进阶要求：打通 iSTFTNet 的动转静和静态图推理流程、复用 PaddleSpeech TTS 已有的静态图推理代码
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
- **技术要求：**
  - 熟悉语音合成模型相关算法。
  - 有基本的模型训练和debug能力。

### No.191：模型复现：JETS <a name='task191'></a>

- **技术标签：Python、深度学习**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - **完成算法复现推理对齐，并按照PaddleSpeech新增算法规范提交PR。**
  - JETS 是一个类似于 VITS 的完全端到端 TTS 模型，请参考论文官方实现，为 csmsc 数据集新增一个 JETS 模型
    - 请最大程度复用 PaddleSpeech FastSpeech2、HiFiGAN 的模块组网代码、数据预处理代码和训练代码
    - 在 https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc 新增 jets example 目录
    - 合成效果优于官方实现（因为 PaddleSpeech 的文本前端优于 JETS 官方实现）
    - 打通 JETS 的动转静和静态图推理流程
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
- **技术要求：**
  - 熟悉语音合成模型相关算法。
  - 有基本的模型训练和debug能力。

### No.192：使用 Gradio 为 PaddleSpeech 语音识别训练过程绘制WebUI工具箱（以conformer模型为例） <a name='task192'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐
- **详细描述：**
  - 使用Gradio 完成语音识别模型的训练过程可视化(conformer + Aishell数据集)
  - 要求至少包含：
    - 数据集校验，检查
    - 训练过程中参数可配置，训练过程可视化
    - 对模型效果进行验证
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
- **技术要求：**
  - 熟悉语音识别模型相关算法。
  - 有基本的模型训练和debug能力。

### No.193：使用 Gradio 为 PaddleSpeech 语音合成声学模型训练过程绘制WebUI工具箱（以fastspeech2模型为例） <a name='task193'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐
- **详细描述：**
  - 使用Gradio 完成语音合成声学模型的训练过程可视化（fastspeech2 + csmsc数据集）
  - 要求至少包含：
    - 数据集校验，检查
    - 训练过程中参数可配置，训练过程可视化
    - 对模型效果进行验证
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
- **技术要求：**
  - 熟悉语音识别模型相关算法。
  - 有基本的模型训练和debug能力。

### No.194：使用 Gradio 为 PaddleSpeech 语音合成声学模型训练过程绘制WebUI工具箱（以conformer模型为例） <a name='task194'></a>

- **技术标签：Python、深度学习**
- **任务难度：基础**⭐
- **详细描述：**
  - 使用Gradio 完成语音合成声码器的训练过程可视化（hifigan + csmsc数据集）
  - 要求至少包含：
    - 数据集校验，检查
    - 训练过程中参数可配置，训练过程可视化
    - 对模型效果进行验证
- **提交内容：**
  - 提交PR包括：代码、模型、推理文档、中英文文档
- **技术要求：**
  - 熟悉语音识别模型相关算法。
  - 有基本的模型训练和debug能力。

### No.195：多学科物理场可视化组件开发 <a name='task195'></a>

- **技术标签：Python，科学计算，可视化组件**
- **任务难度：进阶**⭐⭐
- **详细描述：**
  - 科学计算套件求解多学科物理场，其预测结果具有规范的数据结构，包含1D/2D/3D信息。为对结果进行快速清晰的可视化显示，需要根据预测结果（网络输出数据或保存的中间文件）绘制网络输出变量和用户自定义变量的曲线图、等高线图、矢量图，并提供点/线/面的截取显示和提取等功能。
  - 要求至少包含：
    - 支持稳态和瞬态结果显示，且保存相应图片和动画
    - 1D结果处理：稳态结果用x-r(x)二维曲线图表示；瞬态结果用t-x-r(t, x)二维等高线显示，且支持截取t-r(t, x)和x-r(t, x)绘制二维曲线图
    - 2D结果处理：稳态结果用x-y-r(x, y)二维等高线/矢量图(z多变量时); 瞬态结果中，每一个时刻用x-y-r(x, y)二维等高线/矢量图表示，且支持生成完整动画；稳态结果和瞬态结果均支持截取x-r(x,y)和y-r(x, y)数据绘制二维曲线图
    - 3D结果处理：稳态结果用x-y-z-r(x, y, z)三维等高线/矢量图(z多变量时)表示；瞬态结果中，每一个时刻用x-y-z-r(x, y, z)三维等高线/矢量图表示，且支持生成完整动画；稳态结果和瞬态结果均支持截取任一截面的数据绘制二维等高线/矢量图
    - 支持放大/缩小、切换视图和旋转等操作
- **提交内容：**
  - 提交PR包括：可视化组件设计方案、组件代码、完整demo及测试结果、中英文使用文档
  - 提交Repo地址：[ttps://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S](ttps://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S)
- **技术要求：**
  - 熟练使用Python语言，可使用相关可视化库
  - 了解主流可视化工具（如Tecplot、Paraview）的相关功能和操作过程

### No.196：FVM/FEM/LBM等主流CAE结果提取组件开发 <a name='task196'></a>

- **技术标签：Python，科学计算，CAE，结果提取**
- **任务难度：基础**⭐
- **详细描述：**
  - 科学计算套件采用PINN方法求解高度非线性的多学科物理场时，往往需要采用传统FVM/FEM/LBM等CAE工具来获得监督数据，而传统CAE工具的结果数据具有特定的格式，需要采用定制化的组件来提取所需的监督数据。针对主流CFD软件FLUENT和OpenFoam，CAE结果提取组件读取直接结果文件或导出结果文件，提取或插值指定位置（点/线/面）上的数据，并以PaddleScience接收的格式保存成文件。
  - 要求至少包含：
    - 支持稳态/瞬态结果提以及结构化网格/非结构化网格的结果提取
    - 2D结果提取：根据给定的点/线或时刻，提取或插值点/线/整个区域的稳态或瞬态数据
    - 3D结果提取：根据给定的点/线/面或时刻，提取或插值点/线/面/整个区域的稳态或瞬态数据
    - 完全支持FLUENT和OpenFoam的结果提取
- **提交内容：**
  - 提交PR包括：结果提取组件设计方案、组件代码、完整demo及测试结果、中英文使用文档
  - 提交Repo地址：[ttps://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S](ttps://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S)
- **技术要求：**
  - 熟练使用Python语言以及相关算法
  - 熟悉主流CFD软件FLUENT和OpenFoam保存结果的数据格式

### No.197：论文复现：Robust Regression with Highly Corrupted Data via Physics Informed Neural Networks <a name='task197'></a>

- **技术标签：科学计算**
- **任务难度：基础**⭐️️
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/weipengOO98/robust_pinn
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和PINN网络

### No.198：论文复现：SPNets: Differentiable Fluid Dynamics for Deep Neural Networks <a name='task198'></a>

- **技术标签：科学计算**
- **任务难度：基础**⭐
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/cschenck/SmoothParticleNets
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和CNN网络

### No.199：论文复现：Reduced-order Model for Flows via Neural Ordinary Differential Equations <a name='task199'></a>

- **技术标签：科学计算**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/CarlosJose126/NeuralODE-ROM
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和Reduced-order Model

### No.200：论文复现：An AI-based Domain-Decomposition Non-Intrusive Reduced-Order Model for Extended Domains applied to Multiphase Flow in Pipes <a name='task200'></a>

- **技术标签：科学计算**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/acse-zrw20/DD-GAN-AE
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和Reduced-order Model

### No.201：论文复现：Learning to regularize with a variational autoencoder for hydrologic inverse analysis <a name='task201'></a>

- **技术标签：科学计算**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/madsjulia/RegAE.jl
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和Julia

### No.202：论文复现：Disentangling Generative Factors of Physical Fields Using Variational Autoencoders <a name='task202'></a>

- **技术标签：科学计算**
- **任务难度：基础**️⭐️
- **详细描述：**
  - 在AI Studio中复现论文代码
  - 完成论文中所有案例的精度对齐
  - 参考repo https://github.com/christian-jacobsen/Disentangling-Physical-Fields
- **提交内容：**
  - AI Studio项目
- **技术要求：**
  - 熟练掌握AI Studio、PaddlePaddle和VAE网络

### No.203：论文复现：Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulationsof Airfoil Flows <a name='task203'></a>

- **技术标签：Python，科学计算，论文复现**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**

机翼的“翼型”优化一直是航空领域中分析的重点，尤其是在高雷诺情况下，可能存在的湍流等，目前传统方法是采用RANS、LES等数值计算模型，对于神经网络，我们希望能够将上述数值求解的思路或者泛化性应用于网络模型的定义中。具体任务分为两个阶段，可分阶段完成。

**阶段一：**

- - 基于论文中提供的数据集、翼型设计方法等，基于飞桨框架复现论文原始代码
  - 在跑通论文代码的基础上，能够正确的跑出论文中提出的结果，如文章图8-11所示。

**阶段二（optional，难度大）：**

- - 需要开发者有openfoam或流体相关仿真相关背景
  - 更新openfoam版本至最新版（如9.0以上），并能够复现论文结果
  - 可以实现流场瞬态仿真，并能够替换网络模型为transformer，可模拟瞬态不同工况瞬态特性。
- **提交内容：**
  - 请将代码提交至[AI4S 开源仓库](https://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S)，并在该路径下创建自己的文件夹，格式为对应任务id的taskid/。
  - 优先实现AIStudio项目，针对任务id建立AIStudio项目，若涉及部分工具无法安装，则提供完整的本地代码合入即可
- **技术要求：**
  - 熟练使用Python；
  - 了解UNET、Transformer等网络结构；
  - 简单了解OpenFoam、GMSH等原理。
  - 参考内容：https://arxiv.org/pdf/1810.08217.pdf

### No.204：开放赛题：车辆标准部件受力、变形分析 <a name='task204'></a>

- **技术标签：Python，科学计算，开放赛题**
- **任务难度：进阶**⭐️⭐️
- **详细描述：**

“本任务属于开放性赛题”，围绕汽车、飞机等装备的零部件，进行结构变形、受力分析。本课题选择汽车某标准部件，如下图所示。具体解决两类问题：

- - 能够评估在给定任意的截面外形下，可得到不变的约束、负载条件下的结构受力结果，如应力、形变
  - 能够基于目标的应力、变形，可以逆向优化得到部件截面的理想外形特征

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=fad973b3ca3e4922ad2735a9d3b3cfcc&docGuid=NXFlS7Ad4WBi83)

车辆零部件示意（截面图）

其中提供的数据集生成流程如下：

- - 在给定位移约束和力载荷的情况下，通过改变零部件截面的性质（如改变截面参数化的尺寸），生成该零部件在不同截面下的应力和位移。采用多种不同截面对应的应力、变形结果，从而得到训练数据集
  - 数据集中包含了部件在不同的截面特征参数、复杂、约束条件下对应的各个节点的应力、变形，以及变化后的空间坐标信息，示意如下：

![img](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=3aebcfae590a41deab1fae7a06e02bcc&docGuid=NXFlS7Ad4WBi83)

​       ***note: 数据集会由合作单位产生，无需选手制作***

任务具体描述如下：

- - 如上图所示，黑色线条处施加位移约束（fixed），同时在部件右端施加力载荷
  - 针对同一位移约束和力载荷，给定不同截面参数（如宽度、长度、斜度等参数），生成这个零件的不同截面下的应力和位移
    - 注：上述宽度，长度，斜度等可以作为截面参数化外形的特征参数
  - 结合给定的数据集，自选神经网络构建部件应力、位移预测模型，能够解决任意给定部件截面下的应力、位移预测。
  - 结合给定的数据集，自选神经网络构建部件截面参数特征逆向预测模型，能够解决任意给定目标应力、位移值的前提下，可以得到最优的部件截面特征。

具体任务要求如下：

- - 基于飞桨框架，选择合适的网络模型（如Transformer、GNN等）实现部件变形分析中的正问题与逆问题
- **提交内容：**

请将代码提交至[AI4S 开源仓库](https://github.com/jkrescue/PaddleHackathon4/tree/master/AI4S)，并在该路径下创建自己的文件夹，格式为对应任务id的taskid/。

优先实现AIStudio项目，针对任务id建立AIStudio项目，若涉及部分工具无法安装，则提供完整的本地代码合入即可

- **技术要求：**

熟练使用Python；

了解UNET、Transformer等网络结构；

简单了解结构变形原理





～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～

### 合入标准

-  按 [API 设计规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) 完成 API设计文档；
- 按 [API 验收标准](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) 完成 API功能实现、单测、API文档；
- 稀疏 API 任务需符合稀疏 OP 的特殊开发规范（如有）：
  * 【yaml规则】：写到同一个yaml api，不要写多个，yaml需支持调度
  * 【kernel名规则】：[计算名] + 异构后缀，例如 matmul_csr_dense、softmax_csr、softmax_coo
  * 【文件名规则】：sparse/xx_kernel.cc，sparse/xx_kernel.cu，以目录区分，文件名与dense保持一致

### 参考内容

- [新增API 开发&提交流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_contributing_guides_cn.html)
- [新增 API 设计模板](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)
- [飞桨API Python 端开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_python_api_cn.html)
- [C++ 算子开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)
- [飞桨API文档书写规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_docs_guidelines_cn.html)
- [API单测开发及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)


### 答疑交流

- 如果在开发中对于上述任务有任何问题，欢迎在本 ISSUE 下留言交流。
- 对于开发中的共性问题，在活动过程中，会定期组织答疑，请关注官网&QQ群的通知，及时参与。
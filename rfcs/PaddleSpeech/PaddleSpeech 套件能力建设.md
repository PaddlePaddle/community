# PaddleSpeech 套件能力建设

| 领域             | PaddleSpeech            |
|----------------|-------------------------|
| 提交作者           | Liyulingyue             |
| 提交时间           | 2024-11-08              |
| 版本号            | V1.0                    |
| 依赖飞桨版本         | 3.0.0                   |
| 文件名            | PaddleSpeech 套件能力建设.md  |

# 一、概述

## 1、相关背景

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 paddle.fluid API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。

## 2、功能目标

- 基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 进行适配升级，梳理已有堵点并解决。保证 demo 和 example 目录下已适配的模型在 新 Paddle 版本 & 新其他深度学习框架版本下的正常运转。目前适配版本为 Paddle 2.5.1。
- 基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 中支持转静的模型重新按照 PIR + predictor 的方式导出，并成功推理。

## 3、意义

促进 PaddlePaddle 在语音方向的使用。

# 二、现状

目前 PaddleSpeech 的依赖适配主框架版本位 Paddle 2.5.1，其他依赖库也存在过时的情况，如 librosa 0.8.1等。

飞桨 3.0.0 (beta) 中，对 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等，对 PaddleSpeech 造成了一定的影响。可能会导致部分 API 的调用发生报错，导致模型无法正常训练、推理。

# 三、设计思路与实现方案

针对上述问题，需要对PaddleSpeech 进行适配升级，并保证模型功能与精度不受损失。工作项目如下：

## 1. 对 Demos 的调研与测试
对 Demos 目录下脚本进行测试，验证原有样例功能能够兼容 Paddle 3.0.0-beta 版本。

经测试，Demos 目录下脚本大部分能够兼容 Paddle 3.0.0-beta 版本，总结如下：
- 样例 Metaverse 和 story_talker 的功能与其他套件捆绑，导致依赖复杂，不进行修复。
- 样例 speech_ssl 存在逻辑错误，不是框架升级带来的，已提交PR修复。
- 样例 whisper 在部分分支存在算子不适配的错误，不影响基础功能调用，在后续修复。
- 其他脚本的推理与运行基本正常。

结合 Demos 的测试结果，初步认为 Paddle 3.0.0-beta 版本下，PaddleSpeech 推理功能可以正常使用。

## 2. 对模型的转换
对 PaddleSpeech 中支持转静的模型重新按照 PIR + predictor 的方式导出，并成功推理。

需要转换的模型可以参考[模型列表](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/resource/pretrained_models.py)，具体如下：

| 模型分类                              | 模型名称                           | 版本号  |
|-----------------------------------|-------------------------------|--------|
| asr_static_pretrained_models      |deepspeech2offline_aishell-zh-16k|1.0|
| asr_static_pretrained_models      |deepspeech2online_aishell-zh-16k|1.0.1|
| asr_static_pretrained_models      |deepspeech2online_aishell-zh-16k|1.0.2|
| asr_static_pretrained_models      |deepspeech2online_wenetspeech-zh-16k|1.0.3|
| asr_static_pretrained_models      |deepspeech2online_wenetspeech-zh-16k|1.0.4|
| asr_onnx_pretrained_models        |deepspeech2online_aishell-zh-16k|1.0.2|
| asr_onnx_pretrained_models        |deepspeech2online_wenetspeech-zh-16k|1.0.3|
| asr_onnx_pretrained_models        |deepspeech2online_wenetspeech-zh-16k|1.0.4|
| cls_static_pretrained_models      |panns_cnn6-32k|1.0|
| cls_static_pretrained_models      |panns_cnn10-32k|1.0|
| cls_static_pretrained_models      |panns_cnn14-32k|1.0|
| tts_static_pretrained_models      |speedyspeech_csmsc-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_csmsc-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_ljspeech-en|1.0|
| tts_static_pretrained_models      |fastspeech2_aishell3-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_vctk-en|1.0|
| tts_static_pretrained_models      |fastspeech2_mix-mix|1.0|
| tts_static_pretrained_models      |fastspeech2_mix-mix|2.0|
| tts_static_pretrained_models      |fastspeech2_male-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_male-en|1.0|
| tts_static_pretrained_models      |fastspeech2_male-mix|1.0|
| tts_static_pretrained_models      |fastspeech2_canton-canton|1.0|
| tts_static_pretrained_models      |pwgan_csmsc-zh|1.0|
| tts_static_pretrained_models      |pwgan_ljspeech-en|1.0|
| tts_static_pretrained_models      |pwgan_aishell3-zh|1.0|
| tts_static_pretrained_models      |pwgan_vctk-en|1.0|
| tts_static_pretrained_models      |pwgan_male-zh|1.0|
| tts_static_pretrained_models      |mb_melgan_csmsc-zh|1.0|
| tts_static_pretrained_models      |hifigan_csmsc-zh|1.0|
| tts_static_pretrained_models      |hifigan_ljspeech-en|1.0|
| tts_static_pretrained_models      |hifigan_aishell3-zh|1.0|
| tts_static_pretrained_models      |hifigan_vctk-en|1.0|
| tts_static_pretrained_models      |hifigan_male-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_mix-zh|1.0|
| tts_static_pretrained_models      |fastspeech2_mix-zh|2.0|
| tts_static_pretrained_models      |fastspeech2_mix-en|1.0|
| tts_static_pretrained_models      |fastspeech2_mix-en|2.0|
| tts_static_pretrained_models      |pwgan_male-en|1.0|
| tts_static_pretrained_models      |pwgan_male-mix|1.0|
| tts_static_pretrained_models      |hifigan_male-en|1.0|
| tts_static_pretrained_models      |hifigan_male-mix|1.0|
| tts_static_pretrained_models      |pwgan_aishell3-canton|1.0|
| tts_onnx_pretrained_models        |speedyspeech_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_ljspeech_onnx-en|1.0|
| tts_onnx_pretrained_models        |fastspeech2_aishell3_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_vctk_onnx-en|1.0|
| tts_onnx_pretrained_models        |fastspeech2_cnndecoder_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-mix|1.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-mix|2.0|
| tts_onnx_pretrained_models        |fastspeech2_male_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_male_onnx-en|1.0|
| tts_onnx_pretrained_models        |fastspeech2_male_onnx-mix|1.0|
| tts_onnx_pretrained_models        |fastspeech2_canton_onnx-canton|1.0|
| tts_onnx_pretrained_models        |pwgan_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |pwgan_ljspeech_onnx-en|1.0|
| tts_onnx_pretrained_models        |pwgan_aishell3_onnx-zh|1.0|
| tts_onnx_pretrained_models        |pwgan_vctk_onnx-en|1.0|
| tts_onnx_pretrained_models        |pwgan_male_onnx-zh|1.0|
| tts_onnx_pretrained_models        |mb_melgan_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |hifigan_csmsc_onnx-zh|1.0|
| tts_onnx_pretrained_models        |hifigan_ljspeech_onnx-en|1.0|
| tts_onnx_pretrained_models        |hifigan_aishell3_onnx-zh|1.0|
| tts_onnx_pretrained_models        |hifigan_vctk_onnx-en|1.0|
| tts_onnx_pretrained_models        |hifigan_male_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-zh|1.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-zh|2.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-en|1.0|
| tts_onnx_pretrained_models        |fastspeech2_mix_onnx-en|2.0|
| tts_onnx_pretrained_models        |pwgan_male_onnx-en|1.0|
| tts_onnx_pretrained_models        |pwgan_male_onnx-mix|1.0|
| tts_onnx_pretrained_models        |hifigan_male_onnx-en|1.0|
| tts_onnx_pretrained_models        |hifigan_male_onnx-mix|1.0|
| tts_onnx_pretrained_models        |pwgan_aishell3_onnx-canton|1.0|

## 3. 对重要模型的训练与验证
对 PaddleSpeech 中支持训练的模型进行适配，并验证模型的精度。对于 example 中存在精度信息标注的模型，参考精度进行验证，对于没有精度信息标注的模型，需要在 Paddle 2.5 上进行训练，保留loss后，与Paddle 3.0的训练loss进行对比。

需要适配的训练模型如下：

| 模型名称 | 说明     |
|-----------------------------------|--------|
| conformer |        |
| whisper | 仅推理    |
| pwgan |        |
| fastspeech |        |
| conformer | 流式     |
| wav2vec2 |  |
| hifigan |  |
| vits |  |
| Tacotron2 |  |

## 4. 对其他模型的调通测试
对 PaddleSpeech example 中其他模型运行脚本进行调通测试。保证脚本能正常运行即可。

| 一级目录 | 二级目录 |
|---------|--------|
|aishell|asr1|
|aishell|asr0|
|aishell|asr3|
|vctk|tts3|
|vctk|ernie_sat|
|vctk|vc3|
|vctk|voc1|
|vctk|voc5|
|ljspeech|voc5|
|ljspeech|tts3|
|ljspeech|tts0|
|ljspeech|voc0|
|ljspeech|tts1|
|ljspeech|voc1|
|timit|asr1|
|ted_en_zh|st1|
|ted_en_zh|st0|
|callcenter|asr1|
|tess|cls0|
|aishell3|vits|
|aishell3|vc1|
|aishell3|voc1|
|aishell3|tts3|
|aishell3|vits-vc|
|aishell3|vc0|
|aishell3|voc5|
|aishell3|vc2|
|aishell3|ernie_sat|
|thchs30|align0|
|mustc|st1|
|opencpop|voc1|
|opencpop|voc5|
|opencpop|svs1|
|iwslt2012|punc0|
|tiny|asr0|
|tiny|asr1|
|aishell3_vctk|ernie_sat|
|voxceleb|sv0|
|canton|tts3|
|hey_snips|kws0|
|csmsc|tts0|
|csmsc|jets|
|csmsc|tts3_rhy|
|csmsc|voc5|
|csmsc|voc3|
|csmsc|vits|
|csmsc|tts2|
|csmsc|voc1|
|csmsc|tts3|
|csmsc|voc6|
|csmsc|voc4|
|wenetspeech|asr0|
|wenetspeech|asr1|
|tal_cs|asr1|
|ami|sd0|
|esc50|cls0|
|librispeech|asr4|
|librispeech|asr0|
|librispeech|asr5|
|librispeech|asr3|
|librispeech|asr2|
|librispeech|asr1|
|other|g2p|
|other|tn|
|other|rhy|
|other|augmentation|
|other|cc-cedict|
|other|mfa|
|other|tts_finetune|
|other|ge2e|
|other|punctuation_restoration|
|other|ngram_lm|
|other|spm|
|zh_en_tts|tts3|

# 四、测试和验收的考量

本次任务共三个阶段，验收标准如下：
1. 对于模型动转静，将模型在Paddle 3.0 上进行转换，转换后能够加载模型，并运行推理，成功即可。产出为转换后的模型，如过程中出现问题，应尽可能修复，并留下相关记录。
2. 对于重要模型，需要按照 example 脚本进行训练并对齐精度，产出结果为训练后的模型参数。对于whisper这种仅推理的模型，需在测试集上验证字准。
3. 对于其他模型，需要按照 example 脚本进行进行小批次训练，能够收敛即可。

# 五、排期规划
本次任务已经初步完成Demos的验证工作，对于剩余工作排期如下：
1. 模型动转静，大约11月20日前完成。
2. 模型训练，大约12月10日前完成。
3. 验收测试，大约12月20日前完成。

# 六、影响面

本次任务对 PaddleSpeech 整体功能影响不大，主要为维护性升级。

此文档展示 **PaddlePaddle Hackathon 第八期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情
### 一、PaddleSpeech 套件能力建设 - 模型精度对齐

**任务背景**：

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。外部开发者需要做的事情包括：

**详细描述：**

基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 进行适配升级，梳理已有堵点并解决。保证[example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples) 目录下核心模型在 新 Paddle 版本 & 新其他深度学习框架版本下的正常运转。目前适配版本为 Paddle 2.5.1。

**验收说明：**

PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 10+ 原有模型的适配和精度对齐。

**技术要求：**

- 熟悉 Python，工程能力强
- 对语音识别或合成有一定了解，有训练或者研发经验（加分项）
- 对 PaddleSpeech 套件比较熟悉（加分项）

**参考资料：** https://github.com/PaddlePaddle/PaddleSpeech

### 二、PaddleSpeech 套件能力建设 - PIR导出

**任务背景**：

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。外部开发者需要做的事情包括：

**详细描述：**

基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 中支持转静的模型重新按照 PIR + predictor 的方式导出，并成功推理。

**验收说明：**

PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 20+ 原有静态图模型的重新导出和上传。

**技术要求：**

- 熟悉 Python，工程能力强
- 对语音识别或合成有一定了解，有训练或者研发经验（加分项）
- 对 PaddleSpeech 套件比较熟悉（加分项）

**参考资料：** https://github.com/PaddlePaddle/PaddleSpeech

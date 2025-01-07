# 第十二次 PaddleOCR 开源共建研讨会会议纪要

## 会议概况

- 会议时间：2024-12-04 19:30-20:30
- 会议地点：线上会议
- 参会人：本次会议共有 12 名成员参会，由 PaddleOCR 贡献者组成。本次会议由 [jzhang533](https://github.com/jzhang533) 主持。

## 会议分享与讨论

### 回顾近期的 issue 和 discussions

- 近期关键问题：飞桨框架自 3.0-beta2 后正式启用 PIR， 这会导致模型格式输出为 json 格式，需要有相应的文档说明的同时，接下来需要 PaddleOCR 能够兼容旧模型格式与新模型格式。

### 引入机器人自动回复，示例：<https://github.com/PaddlePaddle/PaddleOCR/issues/14283>

- 经讨论，因机器人的回复可能有误，只适合参考，所以不适宜在 issue 区引入，计划在 discussion 区引入。

### 分享好玩儿的 OCR 项目

- @jzhang533 分享以下项目
  - [manga-ocr](https://github.com/kha-white/manga-ocr), 
  - [MangaOCR](https://github.com/gnurt2041/MangaOCR),
  - [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base)
  - [GOT-OCR](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- 这种采用 visionencoderdecoder 的模型结构的 OCR 项目，适合大家进行学习和探索。
- 会后， @jzhang533 尝试了如何开始一个简单的 visionencoderdecoder 模型，请参考： [demo_ocr](https://github.com/jzhang533/demo_ocr)。

### [PFCCLab/PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/pulls?q=sort%3Aupdated-desc+is%3Apr+is%3Aclosed) 拉新人加入社区

- 会后，汪师傅，邀请了　@BotAndyGao　@yes-github　加入了微信群，非常欢迎。

### 确定下次的部分议题及召开时间

- 时间：2024-12-18
- 轮值主持人：@SWHL

### 轮值值班安排

- 职责：处理 issue 区和 Discussion 区的新增问题，召集，主持例会并记录。
- 期间：每隔两周轮换。
- 排班：@greatv, @jzhang533, @SWHL, @jingsongliujing, @Liyulingyue, @Topdu @mattheliu

序号|GITHUB ID|值班开始|值班结束
:------:|:------:|:------:|:------:
1|@greatv|2024-11-07|2024-11-20
2|@jzhang533|2024-11-21|2024-12-04
3|@SWHL|2024-12-05|2024-12-18
4|@jingsongliujing|2024-12-19|2025-01-01 (放假日) 2025-01-08
5|@Liyulingyue|2025-01-09|2025-01-22 (今年最后一次)
6|@Topdu|2025-02-05|2025-02-19
7|@mattheliu|2025-02-20|2025-03-05

# 第四次 PaddleOCR 开源共建研讨会会议纪要

## 会议概况

- 会议时间：2024-08-14 19:00-20:30
- 会议地点：线上会议
- 参会人：本次会议共有6名成员参会，由 PaddleOCR 贡献者组成。本次会议由[SWHL](https://github.com/SWHL)主持。

## 会议分享与讨论

### 百度工程师 [dyning](https://github.com/dyning) 加入PMC

[dyning](https://github.com/dyning)是百度资深工程师，带领团队从0搭建了PaddleOCR项目，可谓是PaddleOCR元老。他的加入，我相信能让PaddleOCR更上一层楼。

### PaddleOCR 近期进展回顾与 issue 解决进度

- [SWHL](https://github.com/SWHL) 报告回顾了近期 PaddleOCR 的进展，主要为：
    - KIE和RE相关issue解决问题，主要讨论了是否保留这部分，以及是否有其他推荐，暂无形成共识。主要建议有：推荐LLM使用
    - 昇腾设备兼容问题（相关issue [#13647](https://github.com/PaddlePaddle/PaddleOCR/issues/13647)、 [#12436](https://github.com/PaddlePaddle/PaddleOCR/discussions/12436)） → 这边会由[@cuicheng01](https://github.com/cuicheng01)来跟进。
- [SWHL](https://github.com/SWHL) 介绍了 PaddleOCR 文档站点建设的进展，主要更新和修补遗失图像链接，去掉过期链接等
- 本地推理与[在线demo](https://aistudio.baidu.com/community/app/91660)的差异（[Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions?discussions_q=is%3Aopen+label%3Ademo)）。因为在线demo采用的是server的模型，但未明确指出。 → 等[@cuicheng01](https://github.com/cuicheng01)来确定如何解决这一冲突。

### OpenOCR分享

[@Topdu](https://github.com/Topdu)主要分享了所在团队在[OpenOCR](https://github.com/Topdu/OpenOCR)项目中所做的工作，包括复现最新的文本检测和识别算法，提供lbaseline，探索各种论文工作的有效性等等。

OpenOCR是一个很有意义的工作，为广大学者、业界人员提供了平台，希望后期可以与PaddleOCR整合进来，做到学术和业界兼顾。

推荐🔥🔥

### 自由发言，可以提需求给大家讨论、提问等

- [dyning](https://github.com/dyning)讨论了PaddleOCR未来发展，PaddleOCR v5希望有所突破，做到业界第一。
- PaddleOCR项目如何将训练和推理做到合理整合，一是便于吸取最新算法，提供复现baseline；二是提供工业界性能最强、最快、最轻量的部署方案
- doc目录会逐步删除，后续一切文档将站点为主

### 确定下次的部分议题及召开时间

- 下一期由[Topdu](https://github.com/Topdu)担任主席。

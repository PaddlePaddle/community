# 第二次 PaddleOCR 开源共建研讨会会议纪要

## 会议概况
- 会议时间：2024-07-10 19:00-21:00
- 会议地点：线上会议
- 参会人：本次会议共有14名成员参会，由 PaddleOCR 贡献者组成。本次会议由汪昕（[GreatV](https://github.com/GreatV)）主持。

## 会议分享与讨论

### PaddleOCR 近期进展回顾与 issue 解决进度

  * [GreatV](https://github.com/GreatV) 报告了近期 PaddleOCR 的进展，包括：2.8.0 发版；修复多个影响 2.8.0 的 bug；以及近期社区主要关注点。
  * [SWHL](https://github.com/SWHL) 介绍了 PaddleOCR 文档站点建设的进展，并表示近期可以合入到 PaddleOCR 官方仓库。
  * 社区成员讨论多语言支持问题，鼓励用户自己训练对应的语言模型。
  * 社区成员讨论了一些疑难issue，形成意见并回复到相关issue中。

### 会议轮值主席机制讨论：由当期值班人，充当会议主席（负责会议议程安排，会议纪要记录）。

  * 社区成员一致通过了这条提议。下一期由[mattheliu](https://github.com/mattheliu)担任主席。

### 长期任务：PaddleOCR 添加更多 docstring 与 单测 

  * [jzhang533](https://github.com/jzhang533) 提议可以使用 GitHub Copilot 辅助生成 docstring，从而大大减少工作量。

### 短期任务：PaddleOCR numpy 2.0 兼容性

  * [GreatV](https://github.com/GreatV) 和 [jzhang533](https://github.com/jzhang533) 介绍了 imgaug 由于长期未更新，对于 numpy 2.0 缺乏兼容，需要进行替换。

### 自由发言，可以提需求给大家讨论、提问等

  * 社区成员热烈讨论了 PaddleOCR、PaddleX 的发版计划，以及 PaddleX 和 PaddleOCR 联合发版的可能性。
  * 社区成员热烈讨论了 Paddle Inference 当前的问题，包括内存泄漏、CPU推理速度慢等问题。

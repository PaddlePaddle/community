# Paddle Framework Contributor Club 四十八次会议纪要

## 会议概况

- 本次会议时间：2024/11/28 19:00-20:00 (GMT+08:00) 中国标准时间 - 北京
- 本次地点：线上会议
- 参会人：本次会议共有 42 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席刘卓鑫 @[mattheliu](https://github.com/mattheliu) 主持。
- 会议主题：《第四期启航计划/快乐开源专题任务议题讲解》

## 会议纪要

1. 新人介绍：@[MrXnneHang](https://github.com/MrXnneHang)（5 min）
2. 启航计划/快乐开源专题任务介绍
   - [Typo 清理计划](https://github.com/PaddlePaddle/Paddle/issues/69377) @[MrXnneHang](https://github.com/MrXnneHang) (5 min)
   - 任务类型：修复代码中的 typo
   - [CININ编译器后端Pass改造](https://github.com/PaddlePaddle/Paddle/issues/69639) @[Hongqing-work](https://github.com/Hongqing-work) (20 min)
   - 任务类型：后端pass改造
     可参考PR
     [[CINN][Backend Pass Update] Update IfFusion pass #69611](https://github.com/PaddlePaddle/Paddle/pull/69611)
     使用BlockPass改造的合并具有相同条件的连续If的转换函数
     [[CINN]Backend IR and pass refactoring #69454](https://github.com/PaddlePaddle/Paddle/pull/69454)
     可参考其中的stmt_converter/ir_printer进行对stmt类型敏感的定制化访问
   - [PaddleMIX 快乐开源活动](https://github.com/PaddlePaddle/PaddleMIX/issues/787) @[nemonameless](https://github.com/nemonameless) @[luyao-cv](https://github.com/luyao-cv)
     @[cheng221](https://github.com/cheng221) @[WFLiu0327](https://github.com/WFLiu0327) @[yangrongxinuser](https://github.com/yangrongxinuser) (20 min)
   - 任务类型：文档完善及撰写、单测、模型训练及复现
     可参考PR
     [MiniCPM-V 2.6 完善](https://github.com/PaddlePaddle/PaddleMIX/pull/843)
     [LLaVA-OneVision 模型推理](https://github.com/PaddlePaddle/PaddleMIX/pull/796)
     [MiniCPM-V 2.6 推理](https://github.com/PaddlePaddle/PaddleMIX/pull/796)

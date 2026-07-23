# Paddle Framework Contributor Club 六十八次会议纪要

- 本次会议时间：2026/05/21 19:00-21:00 (GMT+08:00) 中国标准时间 - 北京
- 本次会议接入方式：【腾讯会议】
  - 会议密码：无
  - [点击链接入会](https://meeting.tencent.com/dm/2wnqgCogdF72)，或添加至会议列表
- 本次会议主席：[Kozmosa](https://github.com/Kozmosa)
- 本次会议副主席：空缺

## 会议议程
分享环节：
1. 多模态 Agentic RL 稳定高效训练实践 [@Yangruipis](https://github.com/Yangruipis) [@Aurelius](https://github.com/Aurelius)
  - 分享者介绍了小红书多模态Agent RL的业务背景，分析架构升级带来的四大核心挑战。总结了性能调优的四个维度，包括内存、GPU、计算和通信，并给出了并行策略的优化建议。分享者介绍了 Relax 框架的四个特性：孵化容错架构、异步训练流水线、全模态支持和极致性能表现。
2. 多模态 RL 训练探索 [@wangguanzhong](https://github.com/wangguanzhong)
  - 来自百度飞桨的分享者分析了文本环境返回结果拼接、分词、长轨迹奖励分配及延迟效应导致的训推不一致问题。提出TITO方案，通过输入输出传递 token ID，解决多轮交互中多次tokenize不一致的问题。分析训推一致性问题的分类，提出通过检查代码和修正算子解决框架bug。讨论多模态模型在长序列处理中的性能损耗，提出优化方案及并行策略。

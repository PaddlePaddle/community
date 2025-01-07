# Paddle Framework Contributor Club 第三十五次会议纪要

## 会议概况

- 本次会议时间：2024/03/07 19:00-20:00 (GMT+08:00) 中国标准时间 - 北京
- 会议地点：线上会议
- 参会人：本次会议共有 32 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席[MarioLulab](https://github.com/MarioLulab)主持。
- 会议主题：《 Paddle 分布式原理及使用方法、Paddle 分布式的高层 API 调研、黑客马拉松的稿件赛道》


## 会议纪要

1. **新人介绍**：[liujun121533](https://github.com/liujun121533)、[JiehangXie](https://github.com/JiehangXie)、[diadestiny](https://github.com/diadestiny)（5 min）
2. **飞桨资深研发分享** @jeff41404、@lxd-cumt （40 min）
   
    - [分布式 Primer](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/auto_parallel/paddle_distributed_primer.md) 分享 by [jeff41404](https://github.com/jeff41404) （35 min）
      - 分布式训练解决什么问题
      - 常见的分布式原理
      - 怎么使用 Paddle 分布式


    - [飞桨分布式高层 API 需求调研](https://github.com/PaddlePaddle/Paddle/issues/62246) by [lxd-cumt](https://github.com/lxd-cumt) （5 min）
      - 调研用户对分布式使用感受和 API 需求

3. **PFCC 成员自由发言讨论、QA**（10 min）

    - 问题1: 学习分布式训练 or mlsys 的方法？

      答：先动手编程运行，有宏观概念；再深入系统，逐步拆解看论文。

    - 问题2: 支持完全自动并行后, 下一代分布式并行训练架构是什么样的？

      答: 可能是「分布式 + 编译器」or 「分布式 + 国产硬件」
    
    - 问题3: ProcessMesh 的含义? 应该怎么在组网的时候确定?

      答: 描述了进程的笛卡尔拓扑结构，可以在组网时按照不同的分布式策略对 Tensor 进行不同进程上的切分

    - 问题4: 在落地部署场景中, 怎么在不同训练/推理系统之间做资源隔离?

      答: 系统层面上需要 nvidia 的支持, 仍不完善但正在推进；应用层面上目前主流的还是手工隔离

    - 问题5: 当前 Paddle 及其他深度学习框架怎么保证分布式场景中的容灾性?

      答: 目前主流的深度学习框架使用 checkpoints 的方法保存训练过程中的权重, 一旦训练崩溃可以加载 checkpoints 继续训练。目前大部分还不支持与 Hadoop 类似的重新拉起异常计算节点的功能。

4. **黑客松第 6 期新赛道【优秀稿件征集与传播】的预热** by [luotao1](https://github.com/luotao1) （5 min）

    最近1-2年，API数量和功能都发生了非常大的变化。需要让更多的开发者了解这些信息，帮助飞桨传播正面影响力。
    任务形式（分成两阶段）

    - 阶段1：撰写命题制的科技稿件，选手可以写【选定知识点内】的飞桨学习心得、如何使用飞桨等任意有助于扩大飞桨影响力的文章，内含飞桨知识点数量无限制
      
    - 阶段2：以审核通过后的科技稿件为素材，进行影响力传播。

# Paddle Frawework Contributor Club 第二次会议

- 本次会议时间：2022-4-28 19：00
- 本次会议接入方式： 
  - 腾讯会议：476-861-361
  - 会议密码：**** (群内通知)
  - [点击链接入会](https://meeting.tencent.com/dm/0dsJkRjNL5Ow)，或添加至会议列表


- 本次会议主席：高翔（[jeff41404](https://github.com/jeff41404)，飞桨团队）
- 本次会议副主席：骆涛（[luotao1](https://github.com/luotao1)，飞桨团队）

- 本次拟参会Paddle Framework Contributor列表：
    - [jzhang533](https://github.com/jzhang533)
    - [TCChenlong](https://github.com/TCChenlong)
    - [Ligoml](https://github.com/Ligoml)
    - [S-HuaBomb](https://github.com/S-HuaBomb)
    - [guguguzi](https://github.com/guguguzi)
    - [gsq7474741](https://github.com/gsq7474741)
    - [SigureMo](https://github.com/SigureMo)
    - [chenyanlann](https://github.com/chenyanlann)
    - [GeYuYao-hub](https://github.com/GeYuYao-hub)
    - [isLinXu](https://github.com/isLinXu)
    - [yangguohao](https://github.com/yangguohao)
    - [Li-fAngyU](https://github.com/Li-fAngyU)
    - [BrilliantYuKaimin](https://github.com/BrilliantYuKaimin)
    - [jinyouzhi](https://github.com/jinyouzhi)
    - [thunder95](https://github.com/thunder95)
    - [Liyulingyue](https://github.com/Liyulingyue)
    - [xiaoguoguo626807](https://github.com/xiaoguoguo626807)
    - [Asthestarsfalll](https://github.com/Asthestarsfalll)
    - [liqitong-a](https://github.com/liqitong-a)
    - [unseenme](https://github.com/unseenme)
    - [greatv](https://github.com/greatv)
    - [fuqianya](https://github.com/fuqianya)
    - [liyongchao911](https://github.com/liyongchao911)
    - [LielinJiang](https://github.com/LielinJiang)

# 会议议程

- PFCC 简介 & 新同学的一句话自我介绍（5 minutes）
- 集体选出5名在2.3版本中贡献了代码的同学作为代表参加Wave Summit活动。（5minutes）
- 框架贡献者的分享，内容不限，3~4个（30 minutes）
  - [SigureMo](https://github.com/SigureMo)
  - [BrilliantYuKaimin](https://github.com/BrilliantYuKaimin)
  - [Asthestarsfalll](https://github.com/Asthestarsfalll)
- 飞桨接下来会开展的一些重点工作和技术方向介绍（10 minutes），大家可以根据兴趣选择加入开发，每个方向均会配备飞桨在这个领域比较资深的同学一起开发，包括但不限于：
    - API 文档和单测增强：在飞桨开发早期，API 的测试规范没有目前完备，因此有些 API 存在文档描述不够全面或单测缺失的问题。本方向主要面向：增强文档质量，文档能覆盖描述API的所有功能、默认行为的说明和代码一致；增强示例代码，去除示例代码中的旧用法、覆盖API所有参数、更换为更有代表性的例子；增强单测case，补齐API缺失的某些类型的单测case。
    - 性能优化：API 性能是飞桨框架重要功能之一；通过和竞品对比的测试数据，我们发现有些 API 性能不够好，没有充分利用硬件算力；我们内部在不断地进行优化。本方向主要开发：C++ 算子在CPU/GPU 的性能优化，过程中需要和飞桨高性能专家/Intel CPU 专家/NV GPU专家共同交流和探讨，详细请见[算子性能优化Roadmap](https://github.com/PaddlePaddle/Paddle/issues/42286)。
    - 硬件适配：飞桨已经在多款国产训练硬件上全面支持训练和推理任务，并适配了80+模型在曙光 C86加速卡上的运行，以及 30+ 模型在昇腾910芯片上的运行，我们希望进一步扩大这些硬件支持的算子范围以便在这些硬件上支持更多模型训练和推理。本方向主要开发：昇腾910算子适配、海光DCU算子适配等。如果对飞桨国产框架+国产硬件适配感兴趣，可以加入这个方向，深入了解飞桨的硬件适配方案，开发飞桨算子在不同硬件上的实现，并尝试进行算子和模型的性能优化，全面提升端到端的软硬结合领域的实战经验。
    - 扩展数据类型：飞桨现有API已经完整支持float32、float64、int32、int64、bool等常用数据类型，同时部分API支持 fp16、bf16、complex64、complex128 等数据类型，我们希望进一步扩大支持后面这些数据类型的API范围，以实现飞桨API更全面的数据类型支持。本方向主要开发：完善支持 fp16、bf16、complex64、complex128 等数据类型的基础设施（如这些类型的数值微分实现），对还未支持这些数据类型的API开发支持。
- 自由发言，可以提想法、提问和讨论（10 minutes）
- 确定下次会议的主席与副主席，以及召开时间 和 部分议题预告（3 minutes)


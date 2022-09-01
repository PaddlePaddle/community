# Paddle Frawework Contributor Club 第二次会议纪要

## 会议概况

- 会议时间：2022-4-28 19：00 - 20：10
- 会议地点：线上会议
- 参会人：本次会议共有26名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由本次轮值主席高翔([jeff41404](https://github.com/jeff41404))主持。

## 会议分享与讨论

### PFCC 简介 & 两位新成员的一句话自我介绍
考虑有新成员加入，简单回顾了Paddle Frawework Contributor Club（PFCC）PFCC的使命、组织方式和例会安排。
[fuqianya](https://github.com/fuqianya) 和 [liyongchao911](https://github.com/liyongchao911) 两位新成员进行了自我介绍，欢迎加入PFCC！

### 集体选出5名在2.3版本中贡献了代码的同学作为代表参加Wave Summit活动
[候选同学](https://github.com/PaddlePaddle/community/wiki/PFCC-Members-of-PaddlePaddle-2.3)为在2.3版本内有过贡献的同学，PFCC所有成员可进行投票，在4月29号0点前完成。

### 飞桨框架贡献者的议题分享
本次会议，有三位为飞桨框架做出贡献的贡献者向大家介绍了给飞桨框架贡献的经验。
- [SigureMo](https://github.com/SigureMo) ： 向大家介绍了自己给飞桨贡献高层API `paddle.vision.models.MobileNetV3*` 的经历（[#38653](https://github.com/PaddlePaddle/Paddle/pull/38653)）和使用CI及文档预览的一些特性以提升开发效率的经验技巧，在2.3版本内贡献了较多PR！
- [BrilliantYuKaimin](https://github.com/BrilliantYuKaimin)：向大家介绍了自己给飞桨贡献API `paddle.logspace` 的经历([#41261](https://github.com/PaddlePaddle/Paddle/pull/41261)) 及编写单测和文档的经验技巧，很实用！
- [Asthestarsfalll](https://github.com/Asthestarsfalll)：向大家介绍了给飞桨贡献 `paddle.nanquantile` （[#41343](https://github.com/PaddlePaddle/Paddle/pull/41343)）、`paddle.frac`（[#41226](https://github.com/PaddlePaddle/Paddle/pull/41226)）和`paddle.nn.Softmax2D`（[#40910](https://github.com/PaddlePaddle/Paddle/pull/40910)）的经历，及编写代码异常检查、单测和文档的经验，并介绍了使用代码格式检查工具的心得，非常高产！

### 飞桨接下来会开展的一些重点工作和技术方向介绍
为了让大家能 深入地了解飞桨、在飞桨收获更多成长、解决更有挑战性的问题，飞桨团队计划将正在开展的重点工作和技术方向逐渐share给PFCC成员。
本次会议主持人高翔([jeff41404](https://github.com/jeff41404))介绍了飞桨第一批开放的4个技术方向，包括：API文档和单测增强、性能优化、硬件适配 和 扩展数据类型，各方向简介说明参考[会议议程](https://github.com/PaddlePaddle/community/blob/master/pfcc/2022-04-28-meeting-agenda.md)，大家可以根据个人时间和精力选择感兴趣的方向加入，当前性能优化方向已经有[roadmap](https://github.com/PaddlePaddle/Paddle/issues/42286)，其它方向的roadmap会在下次会议前陆续发布。
关于重点工作和技术方向的其它说明：
1. 当前这4个技术方向是第1批，后续会逐步加入其它方向。如果有想开发的方向不在这几个范围，可以在PFCC中反馈
2. 飞桨团队在每个方向都会有工程师支持，和该方向中的同学一起确定目标、规划和分工，希望PFCC的成员能逐渐成为方向骨干甚至是带头人，带领更多人一起开发

### 自由发言和讨论
参会的飞桨框架的贡献者积极的就开发飞桨框架过程中碰到的具体问题，包括开发kernel、数值精度等方面进行了经验分享和交流。
另外，有同学提到希望开发[这个任务](https://github.com/PaddlePaddle/Paddle/issues/40278)，我们下次会请这个任务的工程师进行详细说明。

### 下次会议安排
- 确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：骆涛（[Tao Luo](https://github.com/luotao1)），副主席可自荐或推荐。

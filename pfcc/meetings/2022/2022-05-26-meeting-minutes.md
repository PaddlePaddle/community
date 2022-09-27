# Paddle Frawework Contributor Club 第四次会议纪要



## 会议概况

- 会议时间：2022-5-26 19：00 - 20：00
- 会议地点：线上会议
- 参会人：本次会议共有 23 名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由本次轮值主席石华榜（[S-HuaBomb](https://github.com/S-HuaBomb)，广西大学）主持。

## 会议分享与讨论

### PFCC 简介 & 新成员的一句话自我介绍
PFCC 新成员 [Yulv-git](https://github.com/Yulv-git) 进行了自我介绍，欢迎加入 PFCC！

### GitLink 编程夏令营（GLCC）& PaddlePaddle 活动介绍

- 飞桨团队 [jzhang533](https://github.com/jzhang533) 介绍了 [GitLink](https://www.gitlink.org.cn/glcc)  编程夏令营（GLCC）。GLCC 是在 CCF 中国计算机学会指导下，由 CCF 开源发展委员会（CCF ODC）举办的面向全国高校学生的暑期编程活动。活动将覆盖近千所高校，并联合各大开源基金会、开源企业、开源社区、开源专家，旨在鼓励青年学生通过参加真实的开源软件开发，提升自身技术能力，为开源社区输送优秀人才。
- 本次飞桨 PaddlePaddle 社区开放 4 个项目，项目列表详见 [【Feature Request】CCF GitLink编程夏令营（GLCC） & PaddlePaddle](https://github.com/PaddlePaddle/Paddle/issues/42843)，欢迎对飞桨感兴趣的同学报名参加。

### 飞桨框架 Roadmap API 单测增强、算子性能扩展、算子数据类型扩展专题介绍
- 会议主席 [S-HuaBomb](https://github.com/S-HuaBomb) 对 [PFCC-Roadmap总览](https://github.com/PaddlePaddle/Paddle/issues/42571) 中的 API 单测增强、算子性能扩展 和 扩展数据类型 进行了专题介绍。这 3 个方向有关本次会议的一个议题——API 高维功能扩展，旨在为部分 paddle api 增加高维处理能力。
- 再次鼓励大家可以根据兴趣选择 PFCC-Roadmap 中的各个项目加入开发，每个方向均会配备飞桨在这个领域比较资深的同学一起开发，可点击石墨文档 [链接](https://shimo.im/sheets/RKAWVnVNopC1NKk8/0iaFr) 查看相关 API 详情列表及报名。

### 文档方向启动信息介绍

- 会前，飞桨团队 [TCChenlong](https://github.com/TCChenlong) 向各位 PFCC 成员发送了题为 **【PFCC-Roadmap】飞桨框架文档** 的邮件，介绍了 PFCC - 框架文档方向进一步的规划。邮件中提供了为文档修复和完善做贡献的方案和参考链接。
- 会上，[TCChenlong](https://github.com/TCChenlong) 再次介绍了飞桨框架文档修复方向的背景说明、参与文档修复的做法，并欢迎大家加入 PFCC - 文档方向，我们会每月同步最新的进展与规划，如果大家有任何想法/建议/问题，欢迎随时回复该邮件，与大家一起讨论沟通。

### API 高维功能扩展的议题分享

- 会议主席 [S-HuaBomb](https://github.com/S-HuaBomb) 提了一个API 高维功能扩展新需求 issue [【PFCC-Roadmap】API高维功能扩展 #42904](https://github.com/PaddlePaddle/Paddle/issues/42904)，提出大家可以为部分 paddle api 增加3D输入能力（API范围，增加方式，应用举例，性能考虑等）。会上举例了两个目前在 open 状态的相关 issue：

- `F.affine_grrid` 和 `F.grid_sample()`，目前不支持 3D 处理。

  它们的组合是实现STN（Spatial Transformer Network）的基础组件，常用在图像变形中。增加3D支持对实现可微3D图像（特别是医学图像）变形和配准十分关键。相关issue详见：[【论文复现】paddle的affine_grid如何实现三维功能 · Issue #38670](https://github.com/PaddlePaddle/Paddle/issues/38670)。

- `paddle.logsumexp` 目前不支持 5D Tensor。

  此 OP 计算 x 的指数的总和的对数：logsumexp(x) = log∑exp(x)。常用于对数 softmax 的计算。相关 issue 详见：[【论文复现】paddle.logsumexp不支持5D tensor · Issue #34742](https://github.com/PaddlePaddle/Paddle/issues/34742)。

- 会上，飞桨团队 [jeff41404](https://github.com/jeff41404) 对本议题进行了深入解答，说明了这些需求在飞桨内部已经被关注并讨论过，为类似 API 扩展高维功能涉及到 C/C++ kernel 的编程，但也是可以满足的。

  - 针对 `F.affine_grrid` 和 `F.grid_sample()` 目前不支持 3D 处理的问题，他表示完善这个需求难度不大，相当于黑客松任务的中等偏下难度。
  - 针对 `paddle.logsumexp` 目前不支持 5D Tensor 的问题，他表示这个需求涉及到底层 kernel 的修改，比较复杂，未来会考虑完善类似需求。

### 飞桨框架贡献者的议题分享
本次会议，有两位飞桨框架贡献者向大家分享了经验。

- [fuqianya](https://github.com/fuqianya)：[fuqianya](https://github.com/fuqianya) 有丰富的关于贡献飞桨高层 API、论文复现以及黑客松的经历。分享了为 `paddle.vision.model` 贡献 AlexNet [#36058](https://github.com/PaddlePaddle/Paddle/pull/36058)、SqueezeNet [#36066](https://github.com/PaddlePaddle/Paddle/pull/36066) 和 DenseNet [#36069](https://github.com/PaddlePaddle/Paddle/pull/36069) 等模型、以及为 PaddlePaddle / [PASSL](https://github.com/PaddlePaddle/PASSL) 仓库贡献模型的经验。
- [Yulv-git](https://github.com/Yulv-git)：介绍了他的开源工具 [Search-for-Typos](https://github.com/Yulv-git/Search-for-Typos)，可以一键查找文本或代码中的拼写错误 / 打字错误，对于文档修复工作非常实用。飞桨团队文档修复方向负责人 [TCChenlong](https://github.com/TCChenlong) 和 [Ligoml](https://github.com/Ligoml) 对此非常感兴趣，在线诚邀 [Yulv-git](https://github.com/Yulv-git) 加入文档修复项目。

### 自由发言和讨论
参会的飞桨框架贡献者对 API 高维功能扩展议题、以及两位贡献者的分享进行了交流。

- 对于 API 高维功能扩展，与会者对 `F.grid_sample()` [Issue #38670](https://github.com/PaddlePaddle/Paddle/issues/38670) 中提出的通过 reshape input 使用 API  的 2D 功能处理 3D 输入的方式有性能和精度的担忧，并表示若 pytorch 有相关 API 的高维功能，是否可以搬运代码。飞桨团队 [jeff41404](https://github.com/jeff41404) 表示性能应该不会有问题，精度方面的影响需要实践验证，参考其它开源代码来完善飞桨的 API 是可以的，但需要有自己的创新。
- 与会者对 [Search-for-Typos](https://github.com/Yulv-git/Search-for-Typos) 非常感兴趣，[Yulv-git](https://github.com/Yulv-git) 也希望大家可以加入项目，共同增强工具的功能。

### 下次会议安排
确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：王儒婷（[xiaoguoguo626807](https://github.com/xiaoguoguo626807)），副主席待定。

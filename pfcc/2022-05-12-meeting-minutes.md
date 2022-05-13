# Paddle Frawework Contributor Club 第三次会议纪要

## 会议概况

- 会议时间：2022-5-12 19：00 - 20：10
- 会议地点：线上会议
- 参会人：本次会议共有17名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由本次轮值主席骆涛([Tao Luo](https://github.com/luotao1))主持。

## 会议分享与讨论

### PFCC 简介 & 两位新成员的一句话自我介绍
本次没有新成员参加，此环节跳过。

### 飞桨框架Roadmap和开源之夏活动介绍
- 飞桨团队已公布第一批5个技术方向，包括：API文档和单测增强、性能优化、硬件适配 和 扩展数据类型，详见[PFCC-Roadmap总览](https://github.com/PaddlePaddle/Paddle/issues/42571)，每个方向的Roadmap也已经公布。大家可以根据兴趣选择加入开发，每个方向均会配备飞桨在这个领域比较资深的同学一起开发，可点击[链接](https://shimo.im/sheets/RKAWVnVNopC1NKk8/0iaFr)报名。
- 开源之夏是一项主要面向高校学生的暑期开源活动，旨在鼓励在校学生积极参与开源软件的开发维护，促进优秀开源软件社区的蓬勃发展。本次飞桨团队开放6个项目，欢迎感兴趣的同学报名参加，报名见 [开源之夏 2022 & PaddlePaddle](https://summer-ospp.ac.cn/#/org/orgdetail/824c98a3-c873-4409-871c-db0bf4f272dd/)。

### 飞桨框架的分享
- 飞桨图优化分享（[phlrain](https://github.com/phlrain)）: 介绍本期黑客松 [在 Paddle 中实现 Common Subexpression Elimination（公共子表达式删除）的图优化 pass](https://github.com/PaddlePaddle/Paddle/issues/40278) 任务的基础算法和实现步骤，包括如何写pass、如何实现一个基础算法、如何更新图的结果
- 飞桨高性能算子开发介绍（[JamesLim](https://github.com/JamesLim)）:介绍GPU算子的优化技巧和测试方法，常用的优化技巧有CUDA通用优化技巧、Paddle内置优化技巧、C++模板特性和Paddle内置的cuBlas thrust 等第三方库，测试采用OP Benchmark进行Kernel级性能测试。
- 飞桨phi算子库介绍（[chenwhql](https://github.com/chenwhql) ）：公布了phi算子库的[设计文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design.md)，简单介绍了背景/目标/如何阅读/后续规划，引导开发者上手理解三个PR：[phi的初期设计](https://github.com/PaddlePaddle/Paddle/pull/34425)，[phi + 自定义算子](https://github.com/PaddlePaddle/Paddle/pull/37122)， [yaml生成C++ API初版设计](https://github.com/PaddlePaddle/Paddle/pull/37668)。

### 飞桨框架贡献者的议题分享
本次会议，由为Paddle添加Java Inference（[PR#37162](https://github.com/PaddlePaddle/Paddle/pull/37162)）的同学（[chenyanlann](https://github.com/chenyanlann)）分享。介绍了在 Paddle Inference 支持 C++/C 编译的基础上，通过 JNI 实现 Java 代码与 C、C++ 代码交互，解决Java和Native对象管理和同步、内存管理等问题，提供预测部署的本地 Java 开发体验和功能。

同时，和Inference同学[Shixiaowei02](https://github.com/Shixiaowei02)交流：使用C接口而非直接调用C++接口的原因是，C++编译动态库有问题，C接口具有更好的兼容性。

### 自由发言和讨论
参会的飞桨框架的贡献者积极的就开发飞桨框架过程中碰到的具体问题，包括注册kernel（已提[issue#422370](https://github.com/PaddlePaddle/Paddle/issues/42730)）、支持自定义算子补充高阶导数等方面进行了经验分享和交流。

### 下次会议安排
确定下次会议的时间为两周后的同一个时间段。并确定下次会议的主席为：石华榜（[S-HuaBomb](https://github.com/S-HuaBomb)，广西大学），副主席为王儒婷（[xiaoguoguo626807](https://github.com/xiaoguoguo626807)）。

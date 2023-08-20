# 飞桨开源社区编译 warning 明面消除顺利结项

Hi all：

自2022年8月启动《[开发者联合项目 Call-for-Contribution](../)》以来，编译 warning 消除任务（明面）已经被飞桨开源社区开发者全部认领、并贡献完成。现向各位汇报这个任务的总体进展成果和后续计划。

## 一、项目背景
在大家平时的工作中，经常会有很多很好的想法，但由于缺乏足够的人力、时间、资源或其他原因，其中许多想法尚未实现。 开发者联合项目，会号召一批热爱飞桨、热爱开源的开发者，帮飞桨社区将大家的这些想法变成实现（参考 [Call For Contribution 指引](../guide-to-call-for-contribution_cn.md)）。

作为一个大型的基础设施类开源项目，减少飞桨框架在编译时的warning，可以提升工程质量和整个代码仓库的可维护性，以及减少潜在的bug。Paddle 在 Linux, Mac, Windows 上存在许多 warning，其中大部分是有第三方库引入。去除掉第三方库的影响，以2022年8月16日的CI记录，在 Linux, Mac 上分别存在 40, 743 条明面 warning 记录。此外，还存在着大量未打开的 warning 记录。

## 二、项目历程

整体进展关键时间点如下：

* 2022年10月18日，任务初稿发布，将调研文档整理开放给社区；
* 2022年10月19日，[Tracking Issue]( https://github.com/PaddlePaddle/Paddle/issues/47143 ) 正式建立，开始进行子任务拆分；
* 2022年10月22日，收到并合入了 第一个 PR（来自@Li-fAngyU）；
* 2022年10月29日，去掉 boost 库引入的 warning （来自@GreatV）
* 2023年3月9日，社区基本完成明面 warning 修复，剩余难度较大：Linux 40->9，Mac 743->3；
* 2023年3月31日，@Galaxy1458对剩余难度较大的明面 warning 完成收尾工作。

## 三、项目成果

### 1. 难度适中，吸引开发者踊跃贡献

编译 warning 的消除，是一个了解飞桨框架 C++ 工程实现的非常好的切入点，对于新手非常友好。累计收到了5个社区开发者18个贡献PR，Linux和Mac的明面编译warning全部消除，同时去掉 boost 库引入的 warning，顺利完成任务结项。
* 升级后的编译选项更加严格：
  * 推进方案是对现有的 warning 进行分类修复，修复完一类 warning 后将该类 warning 改为 error 避免 warning 重复出现。
  * 共完成 Linux 5 类、Mac 9 类编译选项的升级，同时去掉 boost 库引入的 5 类编译选项。
* 存量的明面 warning 问题修复：
  * 共修复 Linux 5类40条、Mac 9类740多条的 warning，大幅提升了编译体验，且减少潜在的bug。

### 2. 业内情况对比

编译 warning 的消除会给开发者一个很好的最初印象，使得飞桨框架的代码仓库更加易维护。 Pytorch/Tensorflow 均已消除编译 warning，Paddle经过此项目优化后也消除编译 warning，追平最优竞品开发体验，统计方式如下：
* 使用2023/5/9的编译日志，日志中输入warning关键字，排除第三方库/官方库/以及一些命令的执行结果，只统计来自框架本身的warning
* 日志来源：Pytorch采用[linux-focal-py3.8-gcc7/build](https://github.com/pytorch/pytorch/actions/runs/4921546779/jobs/8791485285?pr=100937#logs)流水线，Tensorflow采用某PR的[build](https://github.com/tensorflow/tensorflow/actions/runs/4630586231/jobs/8236855944?pr=60259)流水线，Paddle采用 [paddle-build](https://xly.bce.baidu.com/paddlepaddle/paddle/newipipe/detail/8391630/job/22689352) 流水线
### 3. 吸引踊跃贡献的人才

来自企业的高水平开发者对编译 warning 消除、代码整洁度更加关注：
* @GreatV 完成了大部分编译 warning 的消除工作，且踊跃参与后续 clang-tidy 的赛题；
* @engineer1109 修复了多个高版本GCC的编译 warning，同时提出了很多建设性的意见。
提升开发者体验，编译是开发者开发的第一步，没有 warning 会给开发者一个很好的最初印象。这几位开发者做的工作助力提升飞桨编译体验和开发体验，从而能吸引更多的开发者。

### 4. 下一步计划

* 未打开的编译 warning 消除：经统计，隐藏的非第三方库引起的编译warning 共计15w左右，Q2计划对隐藏warning的进行分析，推动开启其中必要和有价值的warning编译检查，并完成相应代码修复 @Galaxy1458 
* 更高版本 GCC 的编译 warning 消除：GCC12版本与升级工作同步完成warning消除工作；其他版本GCC通过用户反馈，按需推动修复 @Galaxy1458
* 和社区开发者一起引入 [clang-tidy](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style/code_style_clang_tidy.md)（C++ 代码检查工具），进一步提升飞桨 C++ 代码阅读与开发的便利性。该任务作为 [中国软件开源创新大赛：飞桨框架任务挑战赛](https://github.com/PaddlePaddle/Paddle/issues/53172#paddlepaddle04) 的一个赛题，已收到 RFC 设计文档 （来自 @GreatV）

## 四、致谢
最后，诚挚地感谢社区开发者@Li-fAngyU、@GreatV、@AndPuQing、@engineer1109、 @jinyouzhi 对飞桨框架的积极贡献！

其他对此次飞桨开源社区任务提供建议和帮助的同学，在此无法一一列全。感谢大家持续支持飞桨开源社区的共建工作！

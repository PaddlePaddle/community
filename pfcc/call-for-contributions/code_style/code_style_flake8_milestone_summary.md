# Flake8 代码风格检查工具的引入任务顺利结项

Hi all：

自2022年8月启动《[开发者联合项目 Call-for-Contribution](../)》以来，Flake8 代码风格检查工具的引入任务已经被飞桨开源社区开发者全部认领、并贡献完成。
现向各位汇报这个任务的总体进展成果和后续计划。

## 一、项目背景

在大家平时的工作中，经常会有很多很好的想法，但由于缺乏足够的人力、时间、资源或其他原因，其中许多想法尚未实现。 开发者联合项目） 计划，
会号召一批热爱飞桨、热爱开源的开发者，帮飞桨社区将大家的这些想法变成实现（参考 [Call For Contribution 指引](../guide-to-call-for-contribution_cn.md)）。

Paddle 内部在2022年6月底完成了 pre-commit、pylint、remove-crlf、cpplint、cmake-format、yapf、cmakelint、cmake-format 8 大检查工具的升级，
还剩下两大检查工具 clang-tidy 和 Flake8 还未引入。其中Flake8 是一个被广泛使用的 Python Linter，它利用插件系统集成了多个 Linter 工具，
默认的插件包含了 pycodestyle、pyflakes、mccabe 三个工具，分别用于检查代码风格、语法问题和循环复杂度。此外 Flake8 还支持安装第三方插件，
以对代码进行更全面的检查。因此，在这个背景下发布了《[Flake8 代码风格检查工具的引入](code_style_flake8.md)》任务，期望借助社区开发者的力量帮助飞桨规范代码风格，
提高代码质量，并使开发者能够在开发时发现一些潜在的逻辑问题。

## 二、项目历程

整体进展关键时间点如下：

* 2022年8月25日，任务初稿发布，收到并合入了第一个社区贡献PR（来自@SigureMo）
* 2022年9月14日，[Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/46039) 正式建立，开始进行子任务拆分；
* 2022年10月11日，收到并合入了 [RFC 设计文档](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20220927_introducing_flake8.md)（来自@SigureMo）；
* 2022年10月17日，[Python 代码检查工具 yapf 升级 black 设计方案](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20221018_introducing_black.md)，内部技术评审通过；
* 2022年11月28日，[对 Python 代码引入 import 区域自动重排工具 isort](https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20221111_introducing_isort.md)，内部技术评审通过；
* 2023年2月14日，54个子项全部被贡献完成。

## 三、项目成果
### 1. 难度适中，吸引开发者踊跃贡献
Flake8 代码风格检查工具的引入，是一个了解飞桨框架 Python 工程实现的非常好的切入点，对于新手非常友好。累计收到了4个社区开发者90+个贡献PR（其中2022年国庆期间贡献了50+ PR），修复了4万多行代码的代码风格问题，且完善了过程中发现的多项机制问题，顺利完成任务结项。
* 代码风格检查工具的升级：
  * 引入 flake8 工具用于 python 文件代码检查，版本为4.0.1，规范代码风格，并使开发者能够在开发时发现一些潜在的逻辑问题；
  * 将 Python 代码风格检查工具 yapf 升级为 black，版本为22.8.0，black 相比于 yapf 格式化力度更高，可以自动修复较多的格式问题，大大减少开发者手动解决 flake8 问题的频率；
  * 引入 import 区域自动重排工具 isort，版本为5.11.5，可以极大改善 imports 部分的代码风格，使得开发人员更容易了解模块之间的依赖关系，且能够自动去除 imports 部分重复的导入；
* 存量的代码风格问题修复：
  * Flake8 默认启用的三个工具（pycodestyle、pyflakes、mccabe）共包含了一共 132 个子项。Paddle 中存在问题的子项共 64 个，完成修复54个子项，剩余10个不太方便修复的子项说明如下：
    * 2 个（E203、W503）与 black 不兼容，在使用 black 的项目（包括 pytorch）中基本都会被 ignore 掉。
    * 6 个 （E402、E501、E721、E741、F405、F841）在 pytorch 中也是 ignore 的。https://github.com/pytorch/pytorch/blob/master/.flake8
    * 剩余 2个：E722 和 E731 见[原因](https://github.com/PaddlePaddle/Paddle/pull/50458#issuecomment-1429522203)。
  * 整体上：飞桨共 ignore 10个子项，Pytorch ignore 16个子项，其中重叠有8个，飞桨优于 Pytorch。
* 代码风格检查机制完善：
  * 额外修复 cpplint hook / Remove-CRLF / Detect Private Key hook 在 CI 不生效的问题

### 2. 业内情况参考
Flake8 代码风格检查工具的引入，提升了飞桨 Python代码的质量和可读性。
* pytorch 的 python 代码检查工具为：flake8==3.8.2（飞桨版本4.0.1更新且剩余的错误子项更少，飞桨共 ignore 10个子项，Pytorch ignore 16个子项）, black==22.3.0（飞桨版本22.8.0更新）, mypy==0.950，isort。
* tensorflow 没有使用 flake8、yapf 、black 和 isort，只是推荐手动 yapf 格式化。见issue： [how to auto format python code](https://github.com/tensorflow/tensorflow/issues/50304)，
[Some check that we have enabled in .pylintrc are silently not executed anymore](https://github.com/tensorflow/tensorflow/issues/55442)。

### 下一步计划
后续会考虑和社区开发者一起升级&引入更多的代码风格检查工具，如
[ruff](https://github.com/charliermarsh/ruff)（一个更快的python linter，[前期调研](https://github.com/PaddlePaddle/Paddle/pull/50458#issuecomment-1431280278)已开始）、
[xdoctest](https://github.com/Erotemic/xdoctest) （一个检查文档示例代码的格式化工具）、
解决 [pylint docstring checker 不工作的问题](https://github.com/PaddlePaddle/Paddle/issues/47821) 、
[clang-tidy](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/code_style/code_style_clang_tidy.md)（C++ 代码检查工具），进一步提升飞桨代码阅读与开发的便利性！

## 四、致谢
最后，诚挚地感谢社区开发者@SigureMo、@Yulv-git、@caolonghao、@gglin001, 对飞桨框架的积极贡献！

其他对此次飞桨开源社区任务提供建议和帮助的同学，在此无法一一列全。感谢大家持续支持飞桨开源社区的共建工作！

# Python 2.7/3.5/3.6 旧版本退场任务顺利结项

Hi all：

自2022年8月启动《[开发者联合项目 Call-for-Contribution](../)》以来，Flake8 代码风格检查工具的引入任务已经被飞桨开源社区开发者全部认领、并贡献完成。
现向各位汇报这个任务的总体进展成果和后续计划。

## 一、项目背景

在大家平时的工作中，经常会有很多很好的想法，但由于缺乏足够的人力、时间、资源或其他原因，其中许多想法尚未实现。 开发者联合项目） 计划，
会号召一批热爱飞桨、热爱开源的开发者，帮飞桨社区将大家的这些想法变成实现（参考 [Call For Contribution 指引](../guide-to-call-for-contribution_cn.md)）。

Paddle 从 2.1 版本（2021年）开始不再维护 Python 2.7和3.5，计划从 2.5 版本开始不再维护 Python 3.6，对相关代码进行退场可以提高源码整洁性，
同时开发者可以直接使用python 3 新特性。因此，在这个背景下发布了2个飞桨开源社区任务，
分别是《[Python 2.7 相关代码退场](legacy_python2.md)》和《[Python 3.5/3.6 相关代码退场](legacy_python36minus.md)》，期望借助社区开发者的力量帮助飞桨提高源码整洁性，提升开发者阅读的便利性。

## 二、项目历程

整体进展关键时间点如下：

**《Python 2.7 相关代码退场》任务**：
* 2022年9月27日，任务初稿发布，并收到了第一个社区贡献PR（来自@SigureMo）；
* 2022年10月17日，[Python 2.7 代码退场的设计方案](legacy_python2.md)，内部技术评审通过，并更新任务描述；
* 2022年10月22日，[Tracking Issue 正式建立](https://github.com/PaddlePaddle/Paddle/issues/46837)，完成子任务拆分；
* 2023年1月3日，8个子任务全部被贡献完成；

**《Python 3.5/3.6 相关代码退场》任务**：
* 2022年11月21日，[Python 3.5/3.6 代码退场的设计方案](legacy_python36minus.md)，内部技术评审通过；
* 2022月11月22日，任务发布，并更新 Tracking issue（与上述任务共用一个）；
* 2022年11月29日，收到并合入了第一个社区贡献PR（来自@gsq7474741）；
* 2023年2月6日，3个子任务全部被贡献完成。

## 三、项目成果
### 1. 难度适中，吸引开发者踊跃贡献
Python 旧版本退场，是一个了解飞桨框架 Python 工程实现和 Python 各版本语法特点的非常好的切入点，对于新手非常友好。累计收到了4个社区开发者32个贡献PR，修复代码行数8500+，顺利完成任务结项，其中：
* Python 2.7 相关代码退场，退场了Python 2 子包（`__future__`、`six`）、没有其它功能的 Python 2 模块（类中不必要的显式 object 继承、
super() 函数中不必要的参数、compat.py文件）、Python2 相关逻辑分支、非必要的环境依赖、文档中涉及 Python 2 的内容共5大项8个子项，修复代码行数8000+，
提升开发者阅读的便利性。同时，开发者可以直接使用python 3 新特性，无需考虑和 python 2 的兼容性，更专注于编写代码逻辑。
* Python 3.5/3.6 相关代码退场，在Python 2.7 退场基础上，退场了低于 Python3.7 相关逻辑分支、非必要的环境依赖、文档中涉及 Python 3.5/3.6 的内容共3个子项，
修复代码行数500+，提升开发者阅读和开发源码的便利性。

### 2. 业内情况参考
Python 2.7/3.5/3.6 的退场，使得飞桨框架的代码仓库更加简洁和易维护，也使得飞桨框架跟python编程语言有了更进一步的对齐。以下也列出了业内的情况，供参考。
* Python 2.7 退场：
  * Pytorch 从2020年8月开始逐步清理 Python2.7 代码，见 [Legacy Python2 and early Python3 leftovers](https://github.com/pytorch/pytorch/issues/42919)，
  包括删除six子包、删除`__future__`子包，共2项。
  * Tensorflow 从2022年1月份开始逐步清理 Python 2.7 代码，见 [Cleanup legacy Python2 PR](https://github.com/tensorflow/tensorflow/search?p=2&q=python2%20legacy&type=commits) 列表，
  包含删除six子包、删除__future子包、删除显式object继承、删除super()函数中不必要的参数，共4项。
* Python 3.5/3.6 退场：
  * Pytorch 在1.11版本（2022年3月11日）开始不再维护 Python 3.6，见 [Deprecating Python 3.6 support](https://github.com/pytorch/pytorch/issues/66462)。
  包括下线 Python 3.6 版本的 CI 流水线、不发 Python 3.6 版本的包，但没有看到相关代码退场的 PR。
  * Tensorflow 在2.7.0版本（2021年11月5日）开始不再维护 Python 3.6，见 [tensorflow_tested_build_configurations](https://www.tensorflow.org/install/source#tested_build_configurations)，
  但没有看到 Python 3.5/3.6 代码退场的 PR。

### 下一步计划
后续会考虑和社区开发者一起引入 [pyupgrade](https://github.com/asottile/pyupgrade) （将 python 代码自动化升级到更新版本的工具）或 
[ruff](https://github.com/charliermarsh/ruff)（一个更快的python linter，[前期调研](https://github.com/PaddlePaddle/Paddle/pull/50458#issuecomment-1431280278)已开始），
可自动对旧版本遗留代码进行升级，一方面可以完成旧版本清理任务中未做的「增量控制」，另一方面可以降低未来旧版本清理时的成本（比如 4 个月后即将 EOL 的 Python 3.7），
进一步提升飞桨 Python 代码阅读与开发的便利性！

## 致谢
最后，诚挚地感谢社区开发者@SigureMo、@Yulv-git、@caolonghao、@gsq7474741 对飞桨框架的积极贡献！

其他对此次飞桨开源社区任务提供建议和帮助的同学，在此无法一一列全。感谢大家持续支持飞桨开源社区的共建工作！

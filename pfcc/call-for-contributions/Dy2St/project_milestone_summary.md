# 动转静社区任务顺利结项

**Hi all：**

自2022年11月中旬 PM 联动框架启动《[飞桨快乐开源活动](https://github.com/PaddlePaddle/Paddle/issues/48019)》以来，单机框架动转静方向参与公布了2个任务，已经被飞桨开源社区开发者全部认领、并贡献完成。现向各位汇报这2个任务的总体进展成果和后续计划。

* 飞桨API动静统一：[任务描述](./unify_api_behavior.md)、[Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/48612)
* 动转静功能扩展和接口退场：[任务描述](./to_static_function_extension.md)、[Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/48334)

## 一、项目历程
动转静方向2个飞桨开源社区任务从2022年11月15日正式接受PM邀请发布任务课题，整体进展关键时间点如下：

**《动转静功能扩展和旧接口退场》任务：**

* 11月24日， Tracking Issue 正式建立，完成子任务拆分；
* 11月25日，收到并合入了第一个社区贡献 PR（来自 @SigureMo）
* 2月5日，9个子任务全部被贡献完成，Issue 正式Closed；

**《飞桨API动静行为统一》任务：**

* 12月1日， Tracking Issue 正式建立，完成不统一API任务发布；
* 12月13日， 收到并合入了第一个社区贡献PR（来自 @DrRyanHuang）
* 1月9日，7个子任务全部被贡献完成，Issue 正式 Closed；

## 二、项目成果

**1. 难度适中，吸引高校学生踊跃贡献**

动转静@to_static 功能作为跨越动态图和静态图的一座重要「桥梁」，涉及了飞桨框架动、静态图的技术设计和模块功能，是一个了解飞桨框架整体架构设计的非常好的切入点。且核心接口是在 Python 端实现，对于新手非常友好，很适合由浅入深。
动转静发布的2个飞桨开源社区任务，难度适中，吸引了高校学生的踊跃参与，在2周内完完成了所有子任务的认领，并累计收到了4个社区开发者21个贡献PR，顺利完成任务结项，其中：

* 飞桨API动静统一中，完成了paddle.vision目录下 7个动静不统一API的修复，新增支持静态图；
* 动转静功能扩展任务中，退场了4个陈旧接口，新增了2个辅助功能接口，优化了2个子模块的逻辑，驱动动转静技术治理和易用性提升；

**2. 倾听需求，共建提升飞桨易用性**

在动转静功能扩展任务上，社区开发者同时作为飞桨的真实用户，通过引导从内在需求角度出发，完成了诸如「新增jit.ignore_module接口」、「PartialProgramLayer hash逻辑优化」、「PrintTransformer 优化」等工作，对于飞桨的API生态和用户易用性体验具有重要的意义。

**3. 甄选人才，招揽优秀的开发者**

此次快乐开源活动中有非常多极其活跃、且个人能力优秀的社区开发者，也帮助飞桨团队甄选和招揽了热爱开源的优秀人才，其中也包括即将入职飞桨单机框架的 [@SigureMo](https://github.com/SigureMo) 同学，其曾主导过《flake8 代码风格检查工具的引入》、《Python 2.7 相关代码退场》等社区任务。

后续动转静方向也持续参与《黑客松》项目的任务课题发布，助力飞桨开源社区的蓬勃建设和发展！

## 三、致谢

最后，诚挚地感谢社区开发者 [@DrRyanHuang](https://github.com/DrRyanHuang)、[@Liyulingyue](https://github.com/Liyulingyue)、[@SigureMo](https://github.com/SigureMo)、[@Tomoko-hjf](https://github.com/Tomoko-hjf) 对飞桨框架的积极贡献！


其他对动转静此次飞桨开源社区任务提供建议和帮助的同学，在此无法一一列全。感谢大家持续支持飞桨开源社区的共建工作！

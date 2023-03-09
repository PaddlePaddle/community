# Paddle-TRT算子开发任务顺利结项

**Hi all：**

自2022年11月中旬 PM 联动框架启动《[飞桨快乐开源活动](https://github.com/PaddlePaddle/Paddle/issues/48019)》以来，Paddle-TRT算子开发任务已经被飞桨开源社区开发者全部认领、并贡献完成。现向各位汇报这个任务的总体进展成果和后续计划。

## 一、项目背景

Paddle-TRT 内部在2022年底完成了 TensorRT 算子、功能整体优化升级，在性能与稳定性方面有了整体性提升。Paddle算子体系下有523个算子（包括大粒度算子、小粒度基础算子，去除反向、AMP、分布式、图学习、自动微分等类别），TensorRT提供了44类Layer，包括94个基础算子。目前Paddle主要通过三种机制对TensorRT进行支持：（1）Tensor Layer映射；（2）通用plugin机制（文档参见 [General plugin mechanism](https://github.com/PaddlePaddle/Paddle/pull/45355)）；（3）TensorRT OSS plugin映射，完成TensorRT Layer覆盖和高频算子适配。

由于TensorRT新版本在持续迭代，同时存在使用频次相对较低但重要的算子未能完成适配，需要持续适配开发工作。结合任务开发难度和内部研发计划，与PM一起发布了包括4个TRT算子开发子任务的《[Paddle-TRT 算子开发](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/paddle_trt_optimization.md)》任务，期望借助社区开发者的力量参与Paddle-TRT开发工作，吸引更多开发者参与推理方向工作。

## 二、项目历程

整体进展关键时间点如下：
* 2022年11月22日，任务初稿发布；
* 2022年11月23日，[Tracking Issue](https://github.com/PaddlePaddle/Paddle/issues/48292)正式建立，开始进行子任务拆分；
* 2022年12月08日，收到并合入了第一个社区贡献PR（来自[@zrr1999](https://github.com/zrr1999)）；
* 2023年1月12日，4个子项全部被贡献完成。

## 三、项目成果
**1. 难度适中，吸引开发者踊跃贡献**

Paddle-TRT 是飞桨推理核心功能模块，飞桨训推一体推理环节最重要的一环，也是开发者了解并深入飞桨推理的非常好的切入口。推理方向以Paddle-TRT算子开发任务试水，此次设置了3个简单、1个中等难度子任务，便于开发者熟悉并参与TRT开发贡献工作，从入门到进阶。累计有3位社区开发者，认领全部4个子任务，完成全部任务的PR提交和合入，完成任务结项。有如下特点：
* TensorRT 8.5新特性适配，完成最新发布的IOneHotLayer适配工作，能支持如PP-StructureV2表格识别场景模型性能提升。
* 开发者首次参与算子组合映射适配，TRT无法直接映射算子完成组合映射实现，如reduce_any、reduce_all，有利于开发者参与更多类似算子开发贡献工作。

**2. 优秀社区开发者挖掘**

通过快乐开源活动，发现社区开发者对推理方向开发任务兴趣较高。黑客松上线一周，TensorRT算子任务10个已经完成认领8个，其中有1位开发者[@sanbuphy](https://github.com/sanbuphy) 就来自快乐开源活动。同时该开发者将对前期TRT开发经验进行总结，并进行分享。

**3. 下一步计划**

后续会Paddle-TRT方向会持续参与《[黑客松](https://github.com/PaddlePaddle/Paddle/issues/50629)》项目，发布10个子任务课题，同时推动推理其他子方向任务发布，吸引更多的开发者参与飞桨推理任务开发，助力飞桨开源社区建设与蓬勃发展！

## 四、致谢
最后，诚挚地感谢社区开发者[@DrRyanHuang](https://github.com/DrRyanHuang)、[@sanbuphy](https://github.com/sanbuphy)、[@zrr1999](https://github.com/zrr1999), 对飞桨框架的积极贡献！

其他对此次飞桨开源社区任务提供建议和帮助的同学，在此无法一一列全。感谢大家持续支持飞桨开源社区的共建工作！
# Paddle Framework Contributor Club 四十九次会议纪要

- 本次会议时间：2024/12/12 19:00-20:00 (GMT+08:00) 中国标准时间 - 北京
- 本次地点：线上会议
- 参会人：本次会议共有33名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席肖宇檬@[GoldenStain](https://github.com/GoldenStain)主持。
- 会议主题：《第四期启航计划/快乐开源专题任务议题讲解》
- 录屏链接：https://meeting.tencent.com/crm/KmyzzVBMea

## 会议纪要

1. 新人介绍：@[sunnychenxiwang](https://github.com/sunnychenxiwang)同学和@[zty-king](https://github.com/zty-king)同学（5 min）
2. 启航计划/快乐开源专题任务介绍
   - [Paddle Tensor 规范化（第2期）](https://github.com/PaddlePaddle/Paddle/issues/69908) @[HydrogenSulfate](https://github.com/HydrogenSulfate) (10 min)
		+ 任务类型：适配0-size Tensor与Tensor的规范化
   - [CINN编译器后端Pass注释添加](https://github.com/PaddlePaddle/Paddle/issues/70113) @[Hongqing-work](https://github.com/Hongqing-work) @[gongshaotian](https://github.com/gongshaotian) (20 min)
		+ 任务类型：为后端Pass添加注释，为Pass改造做补充
		+ 可参考PR
			+ [[CINN][Backend Pass Update] Update IfFusion pass #69611](https://github.com/PaddlePaddle/Paddle/pull/69611)
			使用BlockPass合并具有相同条件的连续If的转换函数，并为它们添加严格的注释
3. 开源项目健康度量 Measurement of OSS Projects’ Health @[chyyy510](https://github.com/chyyy510) (25 min)
	- 介绍对开源项目的健康程度进行度量的框架
	- 引入了对不同健康度指标的量化方法
4. Wave Submit 12月26日开发者活动介绍 @[E-Pudding](https://github.com/E-Pudding) (5 min)
	- 介绍Wave Submit线下开发者活动

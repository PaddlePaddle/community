# Pangu-Weather 设计文档

> RFC 文档相关记录信息

|              |                    |
| ------------ | ------------------ |
| 提交作者     | BeingGod          |
| 提交时间     | 2025-03-05         |
| RFC 版本号   | v1.0               |
| 依赖飞桨版本 | 3.0rc1 版本 |
| 文件名       | 20250305_pangu_weather |

## 1. 概述

### 1.1 相关背景

[NO.19 Pangu-Weather 论文复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/【Hackathon_8th】个人挑战赛—套件开发任务合集.md#no19-pangu-weather-论文复现)

盘古气象大模型(Pang-Weather)是首个精度超过传统数值预报方法的 AI 方法，其提供了 1 小时间隔、3 小时间隔、6 小时间隔、24 小时间隔的预训练模型。1 小时 - 7 天预测精度均高于传统数值方法（即欧洲气象中心的 operational IFS），同时预测速度提升 10000 倍，可秒级完成对全球气象的预测，包括位势、湿度、风速、温度、海平面气压等。盘古气象大模型的水平空间分辨率达到 0.25°×0.25° ，时间分辨率为 1 小时，覆盖 13 层垂直高度，可以精准地预测细粒度气象特征。

### 1.2 功能目标

1. 复现 Pangu-Weather 代码，实现完整的推理流程。
2. 保持精度与作者原代码一致，相对误差在 ±10% 以内。
3. 产出论文相关文档、图片等。

### 1.3 意义

复现 Pangu-Weather 代码，能够使用 Pangu-Weather 模型进行推理。

## 2. PaddleScience 现状

PaddleScience 套件暂无 Pangu-Weather 代码案例。

## 3. 目标调研

- 论文解决的问题：
  Pangu-Weather 模型解决了以往天气预报效率低的问题
- 链接：
  1. 代码：[https://github.com/198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather)
  2. 论文：[https://arxiv.org/abs/2211.02556](https://arxiv.org/abs/2211.02556)

## 4. 设计思路与实现方案

参考 PaddleScience 已有代码实现 Pangu-Weather

1. 基于 Predictor 类实现推理代码
2. 实现推理结果可视化

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

推理结果可正常可视化；精度与作者原始代码对齐。

## 6. 可行性分析和排期规划

- 2025.3上旬：调研，复现代码并作调整
- 2025.3中旬：整理项目产出，撰写案例文档

## 7. 影响面

丰富 PaddleScience 的应用案例，在 examples 新增 pangu-weather 推理案例
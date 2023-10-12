# 复现 PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for Solving Parameterized Steady-State PDEs on Irregular Domain 的 case0 和 case2 案例

|              |                            |
| ------------ | -------------------------- |
| 提交作者     | lizechng                   |
| 提交时间     | 2023-10-12                 |
| RFC 版本号   | v1.0                       |
| 依赖飞桨版本 | develop/release x.x.x 版本 |
| 文件名       | 20231012_phygeonet.md      |

## 1. 概述

### 1.1 相关背景

- 【PaddlePaddle Hackathon 5th】开源贡献个人挑战赛科学计算任务合集 [No.60：PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for Solving Parameterized Steady-State PDEs on Irregular Domain](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/【PaddlePaddle%20Hackathon%205th】开源贡献个人挑战赛科学计算任务合集.md#no60phygeonet-physics-informed-geometry-adaptive-convolutional-neural-networks-for-solving-parameterized-steady-state-pdes-on-irregular-domain)

### 1.2 功能目标

- 复现 PhyGeoNet: Physics-Informed Geometry-Adaptive Convolutional Neural Networks for Solving Parameterized Steady-State PDEs on Irregular Domain 的 case0 和 case2 案例。

### 1.3 意义

- 复现 PhyGeoNet 并完成 case0 和 case2 相关案例。

## 2. PaddleScience 现状

- PaddleScience 套件暂无 PhyGeoNet 模型案例。

## 3. 目标调研

- **论文链接：** https://arxiv.org/abs/2004.13145

- **参考代码链接：** https://github.com/Jianxun-Wang/phygeonet

- 可以参考已有代码和论文相关描述进行复现。

## 4. 设计思路与实现方案

- 参考已有代码实现 PhyGeoNet.

### 4.1 补充说明[可选]

- 无。

## 5. 测试和验收的考量

- 复现达到原有代码精度。

## 6. 可行性分析和排期规划

- 参考代码修改为 paddle 实现，使用PaddleScience API，测试精度对齐
  - 2023年10月中旬完成 case0 案例
  - 2023年10月下旬完成 case2 案例

## 7. 影响面

- 为 PaddleScience 增加 PhyGeoNet 案例

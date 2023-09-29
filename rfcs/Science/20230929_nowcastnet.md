# PhyCRNet 设计文档

|              |                    |
| ------------ | -----------------  |
| 提交作者      |      co63oc              |
| 提交时间      |       2023-09-29   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop/release 2.5.0 版本        |
| 文件名        | 20230929_nowcastnet.md             |

## 1. 概述

### 1.1 相关背景

[No.61：Skillful nowcasting of extreme precipitation with NowcastNet](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no61skillful-nowcasting-of-extreme-precipitation-with-nowcastnet)

### 1.2 功能目标

复现NowcastNet模型 mrms_case和mrms_large_case，并合入PaddleScience

### 1.3 意义

复现NowcastNet模型 mrms_case和mrms_large_case，并合入PaddleScience

## 2. PaddleScience 现状

PaddleScience 套件暂无相关模型案例。

## 3. 目标调研

参考代码 https://codeocean.com/capsule/3935105/tree/v1
论文链接 https://www.nature.com/articles/s41586-023-06184-4#Abs1

## 4. 设计思路与实现方案

参考已有代码实现复现NowcastNet模型

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

复现达到原有代码精度

## 6. 可行性分析和排期规划

参考代码修改为 paddle 实现，使用PaddleScience API，测试精度对齐

## 7. 影响面

为 PaddleScience 增加 NowcastNet 模型案例

# 任务名

> RFC 文档相关记录信息

|              |                       |
| ------------ | --------------------- |
| 提交作者     | co63oc                |
| 提交时间     | 2024-11-05            |
| RFC 版本号   | v1.0                  |
| 依赖飞桨版本 | develop版本           |
| 文件名       | 20241105_harmonics.md |

## 1. 概述

### 1.1 相关背景

torch-harmonics 是一个基于 PyTorch 的库用于实现球谐函数(spherical harmonics)相关的操作和计算。球谐函数在物理学、计算机图形学、以及许多其他领域中有着广泛的应用，尤其是在处理三维数据(如立体图像处理、分子建模、天文学等)时非常有用。

飞桨适配 torch-harmonics https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5

### 1.2 功能目标

1.整理 torch-harmonics 的所有公开 API
2.使用 paddle 的 python API 等价组合实现上述公开 API 的功能
3.参考 pytorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过

### 1.3 意义

可以在飞桨中使用 torch-harmonics 的API调用 

## 2. PaddleScience 现状

PaddleScience 暂无 torch-harmonics 的API调用

## 3. 目标调研

参考 torch-harmonics 文档，已有模块包含：
Spherical harmonics
Spherical harmonic transform
Discrete Legendre transform
Solving the Helmholtz equation
Solving the shallow water equations

## 4. 设计思路与实现方案

参考torch 和 paddle 的API映射文档
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html
转换部分API，不能适配规则转换的API修改适配，包含转换工具消除的注释、类型提示、换行等部分。

### 4.1 补充说明[可选]


## 5. 测试和验收的考量

编写飞桨后端的单测文件测试

## 6. 可行性分析和排期规划

2024.11 RFC文档
2024.11 提交PR修改代码

## 7. 影响面

PaddleScience 增加 harmonics 的接口支持

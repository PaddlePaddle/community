# 任务名

> RFC 文档相关记录信息

|              |                       |
| ------------ | --------------------- |
| 提交作者     | beinggod |
| 提交时间     | 2024-11-11            |
| RFC 版本号   | v1.0                  |
| 依赖飞桨版本 | develop版本           |
| 文件名       | 20241111_harmonics.md |

## 1. 概述

### 1.1 相关背景

torch-harmonics 是一个基于 PyTorch 的库用于实现球谐函数(spherical harmonics)相关的操作和计算。球谐函数在物理学、计算机图形学、以及许多其他领域中有着广泛的应用，尤其是在处理三维数据(如立体图像处理、分子建模、天文学等)时非常有用。

飞桨适配 torch-harmonics https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5

### 1.2 功能目标

全量支持 torch-harmonics 对应的 API 调用。


### 1.3 意义

PaddleScience 添加 harmonics 中对应 API，丰富 PaddleScience 案例。

## 2. PaddleScience 现状

PaddleScience/ppsci/arch 目录下存在部分 harmonics 适配文件，经评估暂不影响本次适配工作。

## 3. 目标调研

适配目标主要包括 3 部分
1. torch-harmonics/torch_harmonics 下核心模块（包括分布式相关功能）
2. torch-harmonics/tests 下单测
3. torch-harmonics/notebooks 下案例

### 3.1 torch_harmonics 核心 API

1. Spherical harmonics (包含分布式相关功能)
2. Spherical harmonic transform (包含分布式相关功能)
3. Discrete Legendre transform
4. Solving the Helmholtz equation
5. Solving the shallow water equations

### 3.2 tests 单测

1. test_sht.py
2. test_convolution.py
3. test_distribution.py

### 3.3 notebooks 案例

1. [Getting started](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/getting_started.ipynb)
2. [Quadrature](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/quadrature.ipynb)
3. [Visualizing the spherical harmonics](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/plot_spherical_harmonics.ipynb)
4. [Spectral fitting vs. SHT](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/gradient_analysis.ipynb)
5. [Conditioning of the Gramian](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/conditioning_sht.ipynb)
6. [Solving the Helmholtz equation](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/helmholtz.ipynb)
7. [Solving the shallow water equations](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/shallow_water_equations.ipynb)
8. [Training Spherical Fourier Neural Operators](https://github.com/NVIDIA/torch-harmonics/tree/v0.6.5/notebooks/train_sfno.ipynb)


## 4. 设计思路与实现方案

参考 Paddle 与 PyTorch API 转换文档，将 PyTorch 中对应的 API进行改写。 

### 4.1 技术难点

1. paddle 不支持 complex 类型参数，需要使用两个tensor进行模拟
2. paddle 分布式通信 API 不支持 complex 类型，需要使用两个tensor进行模拟
3. sht 分布式场景下会遇到数据不均等切分问题，paddle 分布式通信 API 会进行拦截，需要进行 padding
4. tensorly 暂未提供 tensorly-paddle 支持，需要对依赖的模块进行单独实现


## 5. 测试和验收的考量

1. 通过 PaddleScience 的代码风格检查
2. 全量 notebooks 示例跑通
3. 全量单测测试通过

## 6. 可行性分析和排期规划

1. 提交RFC 11月
2. 完成PR合入 11月

## 7. 影响面

PaddleScience 增加 harmonics 的接口支持

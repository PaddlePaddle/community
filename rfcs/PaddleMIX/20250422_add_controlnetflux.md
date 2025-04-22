
# Flux ControlNet 模型复现

|              |                    |
| ------------ | -----------------  |
| 提交作者      |     co63oc               |
| 提交时间      |       2025-04-22   |
| RFC 版本号    | v1.0               |
| 文件名        | 20250422_add_controlnetflux.md             |

## 1. 概述

### 1.1 相关背景

> [NO.34 Flux ControlNet 模型复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_8th/%E3%80%90Hackathon_8th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E5%A5%97%E4%BB%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no34-flux-controlnet-%E6%A8%A1%E5%9E%8B%E5%A4%8D%E7%8E%B0)

### 1.2 功能目标

> 具体需要实现的功能点。

1. 在 PaddleMIX 套件 PPDiffusers 模块中实现模型组网和相关推理 Pipeline
2. 在 PaddleMIX 套件中新增对应单测

### 1.3 意义

在 PaddleMIX 套件中复现 Flux ControlNet 模型。

## 2.  方案背景

diffusers中已有 Flux ControlNet 模型的实现，可以直接在 PaddleMIX 套件中复现。

## 3. 目标调研

diffusers 中模型文件 src/diffusers/models/controlnets/controlnet_flux.py
diffusers 中 pipelines 目录 src/diffusers/pipelines/flux/
diffusers 中单测目录 tests/pipelines/controlnet_flux/

## 4. 设计思路与实现方案

对照PyTorch 最新 release 与 Paddle develop API 映射表 https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html，转换代码中的api调用，然后比较对齐精度

### 4.1 补充说明[可选]

无

## 5. 测试和验收的考量

在 PaddleMIX 套件中新增对应单测，并保证精度对齐。

## 6. 可行性分析和排期规划

2025-04 创建提交RFC文档
2025-05 实现具体代码

## 7. 影响面

在 PaddleMIX 套件中新增 Flux ControlNet 模型。


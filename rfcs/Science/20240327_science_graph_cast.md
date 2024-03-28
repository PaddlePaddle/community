# GraphCastNet 代码迁移至 PaddleScience

|              |                    |
| ------------ | -----------------  |
| 提交作者      |   MayYouBeProsperous  |
| 提交时间      |       2024-03-27   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | 2.6 版本        |
| 文件名        | 20240327_science_graph_cast.md  |

## 1. 概述

### 1.1 相关背景

GraphCast 这种方法是天气预报领域的一项重大进展，它利用机器学习的能力来提高预测的准确性和效率。GraphCast 通过图神经网络（GNNs）建模复杂的天气动态，并在欧洲中期天气预报中心（ECMWF）的ERA5再分析数据集上进行训练。它在全球范围内以0.25°的高分辨率快速预测数百种天气变量，并在多项目标上超越了ECMWF的高分辨率预测系统（HRES）。这项研究表明，GraphCast不仅能提高标准天气预测的效率，还在预测严重天气事件方面显示出潜力，可能对依赖天气的决策过程产生重大影响。

### 1.2 功能目标

使用 PaddleScience 套件复现 GraphCast 案例，并使推理过程误差满足要求。

### 1.3 意义

丰富 PaddleScience 套件功能，完善套件在地球科学方向的应用案例。

## 2. PaddleScience 现状

PaddleScience 套件中有完善的套件模块，比如数据加载、网络架构、优化器和求解器等，能够很便捷地构建新的模型。目前套件中已有 [GraphCast 案例](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/graphcast)，但是案例基于 Paddle 的实现，未使用 PaddleScience API 实现。

## 3. 目标调研

[PR 699](https://github.com/PaddlePaddle/PaddleScience/pull/699) 使用 Paddle 复现了 GraphCast 案例，本次任务将在此基础上，使用 PaddleScience API 实现把 GraphCast 案例。

可能存在的难点如下：

1. 数据集加载和图的构建过程。

2. 将模型接入到 PaddleScience 模型架构。

## 4. 设计思路与实现方案

1. 预训练模型加载

制作 PaddleScience 可用的预训练模型。

2. 数据集加载

在 `ppsci.data.dataset` 中实现一个新的数据加载类 `GraphGridMeshDataset`，读取数据集并构建图结构。
```
dataloader_cfg = {
    "dataset": {
        "name": "GraphGridMeshDataset",
        "file_path": cfg.DATASET_PATH,
        "input_keys": ... ,
        "label_keys": ... ,
        "alias_dict": ... ,
    },
}
```
3. 模型构建

在 `ppsci.arc` 中实现 `GraphCastNet` 模型，并用以下形式调用模型。

```python
model = ppsci.arch.GraphCastNet
```

4. 评估器构建

根据评价指标实现损失函数，构建评估器。

```python
sup_validator = ppsci.validate.SupervisedValidator(
    eval_dataloader_cfg,
    ppsci.loss.FuntionalLoss( ... ),
    ...
)
validator = {sup_validator.name: sup_validator}
```

5. 模型评估推理

构建 `Solver`，开始训练评估推理。

```python
solver = ppsci.solver.Solver( ... )
solver.eval()
solver.predict( ... )
```

6. 可视化

使用 plot 进行可视化，原案例已经有可视化代码。

## 5. 测试和验收的考量

实验复现精度与[原案例](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/graphcast)保持一致，推理过程误差在1e-5以下。


## 6. 可行性分析和排期规划

2024.04.01~2024.05.07 完成案例代码的编写和调试。

2024.05.08~2024.05.15 完成案例文档的编写。

## 7. 影响面
在`ppsci.data.dataset` 模块中增加数据加载类，在`ppsci.arch` 模块中增加新模型。

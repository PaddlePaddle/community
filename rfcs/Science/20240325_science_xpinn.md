# XPINN 迁移至 PaddleScience

|              |                    |
| ------------ | -----------------  |
| 提交作者      |   MayYouBeProsperous  |
| 提交时间      |       2024-03-25   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | 2.6 版本        |
| 文件名        | 20240325_science_xpinn.md  |

## 1. 概述

### 1.1 相关背景

求解偏微分方程(PDE) 是一类基础的物理问题，随着人工智能技术的高速发展，利用深度学习求解偏微分方程成为新的研究趋势。PINNs(Physics-informed neural networks) 是一种加入物理约束的深度学习网络，因此与纯数据驱动的神经网络学习相比，PINNs 可以用更少的数据样本学习到更具泛化能力的模型，其应用范围包括但不限于流体力学、热传导、电磁场、量子力学等领域。

基于 PINNs 框架，A.D.Jagtap 等人提出了一种扩展 PINNs —— XPINNs 。对比传统的 PINNs，XPINNs 有如下特点：

1. 广义时空域分解： XPINN 提供在 $C^0$ 或者更规则的边界上的高度不规则的、凸与非凸的时空域分解。

2. 可扩展到任意的微分方程：基于 XPINN 的方法的域分解方法可以扩展到任何类型的偏微分方程。

3. 简单的界面条件：XPINN可以轻松扩展到任何复杂的几何形状，以及更高的维度上。

### 1.2 功能目标

使用 PaddleScience 套件复现 XPINN 案例，并使训练精度满足要求。

### 1.3 意义

丰富 PaddleScience 套件内容，验证套件在求解 PDE 问题上应用的正确性和广泛性。

## 2. PaddleScience 现状

PaddleScience 套件中有完善的套件模块，比如数据加载、网络架构、优化器和求解器等，能够很便捷地构建新的模型。目前套件中已有 [XPINN 案例](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/XPINNs)，但是案例基于 Paddle 的实现，未使用 PaddleScience API 实现。

## 3. 目标调研

[PR 535](https://github.com/PaddlePaddle/community/pull/535) 使用 Paddle 复现了 XPINN 案例，本次任务将在此基础上，使用 PaddleScience API 实现把 XPINN 案例，总体难度不大。

可能存在的难点是 XPINN 模型代码需要做较多修改才能接入 PaddleScience 模型架构中。

## 4. 设计思路与实现方案

实验案例是二维泊松方程的求解：

$$ \Delta u = f(x, y), x,y \in \Omega \subset R^2$$

边界表达式：

$$ r =1.5+0.14 sin(4θ)+0.12 cos(6θ)+0.09 cos(5θ) $$

上述区域被分为三个不规则的、非凸的子域训练求解，数据集中包含三个子域的数据点。


1. 数据集加载

数据集以字典的形式存储在 .mat 文件中，使用 `IterableMatDataset` 读取数据。

```python
train_dataloader_cfg = {
    "dataset": {
        "name": "IterableMatDataset",
        "file_path": cfg.DATASET_PATH,
        "input_keys": ... ,
        "label_keys": ... ,
        "alias_dict": ... ,
    },
}
```

2. 模型构建

在 `ppsci.arc` 中实现 `XPINN` 模型，并用以下形式调用模型。

```python
model = ppsci.arch.XPINN
```

3. 参数和超参数设定

与原案例相同，学习率设为 0.0008，训练轮数设为 501。

4. 优化器构建

优化器使用 Adam 优化器。
```python
optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)
```

5. 约束构建

使用监督约束 `SupervisedConstraint` 构建约束，损失函数需要自行定义。

```python
sup_constraint = ppsci.constraint.SupervisedConstraint(
    train_dataloader_cfg,
    ppsci.loss.MSELoss( ... ),
    ...
)
constraint = {sup_constraint.name: sup_constraint}
```

6. 评估器构建

评价指标 metric 为 L2 正则化函数。
```python
sup_validator = ppsci.validate.SupervisedValidator(
    eval_dataloader_cfg,
    ppsci.loss.MSELoss( ... ),
    ...
)
validator = {sup_validator.name: sup_validator}
```

7. 模型训练、评估

构建 `Solver`，开始训练评估。

```python
solver = ppsci.solver.Solver( ... )
# train model
solver.train()
# evaluate after finished training
solver.eval()
```

8. 可视化

使用 plot 进行可视化，原案例已经有可视化代码。

## 5. 测试和验收的考量

实验复现精度与 paddle 一致：

|	  |	paddle|
|---|---|
|Test Loss	| 2.138322e-01 |

## 6. 可行性分析和排期规划

2024.03.25~2024.04.30 完成案例代码的编写和模型训练。

2024.05.01~2024.05.07 完成案例文档的编写。

## 7. 影响面

在`ppsci.arch` 模块中增加新模型。

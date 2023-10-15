# Science 53 设计文档 

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |DUCH714             |
| 提交时间      |2023-10-16          |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        |hackthon5th_53.md   |

## 1. 概述

### 1.1 相关背景

> 最近几年，深度学习在很多领域取得了非凡的成就，尤其是计算机视觉和自然语言处理方面，而受启发于深度学习的快速发展，基于深度学习强大的函数逼近能力，神经网络在科学计算领域也取得了成功，现阶段的研究主要分为两大类，一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Retz Net，另一类是通过数据驱动的深度神经网络算子，其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用，比如天气预测，量子化学，生物工程，以及计算流体等领域。

### 1.2 功能目标

> 在本任务中，作者根据[NSFnets (Navier-Stokes Flow nets): Physics-informed
neural networks for the incompressible Navier-Stokes
equations](https://arxiv.org/abs/2003.06496)中的代码[NSFNet code](https://github.com/Alexzihaohu/NSFnets/blob/master/)
>
> 复现Kovasznay flow + cylinder wake + Beltrami flow，三个案例和指标。

### 1.3 意义

> 基于以paddle为backend的[DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html)框架实现纳韦斯托克方程问题的高精度求解，包括时间独立的二维纳韦斯托克方程(Kovasznay flow)，时间依赖的二维纳韦斯托克方程(cylinder wake)以及时间依赖的三维纳韦斯托克方程(Beltrami flow)，为求解其它流体问题提供样板。

## 2. PaddleScience 现状

> PaddleScience 无法完成训练，从[AIstudio](https://aistudio.baidu.com/studio/project/partial/verify/6832363/fa46b783a28442b88fb7d2756ffddb6c)中的后台任务_nfs1_可以看到，速度的相对误差在O(10^(-2))附近，因此转而使用基于paddle框架的DeepXDE。

## 3. 目标调研

>论文对纳韦斯托克方程进行了高精度求解，其难点主要在于代码较为陈旧，为基于原PINN而编写的tensorflow v1.0 版本代码，同时训练数据集以及训练网络庞大，单核GPU难以完成相关训练。

## 4. 设计思路与实现方案

> 基于PaddleScience API实现求解2D稳态热传导方程的设计思路与实现步骤如下：

1、模型构建

2、方程构建

3、计算域构建

4、约束构建

5、超参数设定

6、优化器构建

7、评估器构建

8、可视化器构建

9、**算力卡的囤积**

10、模型训练、评估与可视化

### 4.1 补充说明

> 1、由于算力原因，本文没有使用文中所提及的 100×10 层MLP网络，而是使用 50×5 的MLP,同时训练点也进行大幅削减。
> 2、对于压力p由于会出现与原始数据偏移一个常数C的问题，我们在计算相对误差是会先减去该常数C。

## 5. 测试和验收的考量

> 精度对齐
> **Kovasznay flow**

| alpha=1 size 4*50 | paper  | code(without BFGS) | paddle(DeepXDE)  |
|-------------------|--------|--------------------|---------|
| u                 | 0.084% | 0.062%             | 0.015%  |
| v                 | 0.425% | 0.431%             | 0.077%  |
| p                 | 0.309% | /                  | 0.038%  |

>**Cylinder wake**

| alpha=1 size 4*50 | paper | code(without BFGS) | paddle (DeepXDE) |
|-------------------|-------|--------------------|------------------|
| u                 | /     | 0.269              | 0.011            |
| v                 | /     | 0.985              | 0.047            |
| p                 | /     | /                  | 0.818            |
## 6. 可行性分析和排期规划

>202309 :  调研

>202310 ：基于TF以及DeepXDE的复现

>202311 ：基于PaddleScience的复现

>202312 ：整理项目产出，撰写案例文档

## 7. 影响面

> 帮助PaddleScience实现纳韦斯托克方程的高精度复现

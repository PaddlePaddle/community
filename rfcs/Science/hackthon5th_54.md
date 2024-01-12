# Science 54 设计文档 

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |DUCH714             |
| 提交时间      |2023-10-16          |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        |hackthon5th_54.md   |

## 1. 概述

### 1.1 相关背景

> 最近几年，深度学习在很多领域取得了非凡的成就，尤其是计算机视觉和自然语言处理方面，而受启发于深度学习的快速发展，基于深度学习强大的函数逼近能力，神经网络在科学计算领域也取得了成功，现阶段的研究主要分为两大类，一类是将物理信息以及物理限制加入损失函数来对神经网络进行训练, 其代表有 PINN 以及 Deep Retz Net，另一类是通过数据驱动的深度神经网络算子，其代表有 FNO 以及 DeepONet。这些方法都在科学实践中获得了广泛应用，比如天气预测，量子化学，生物工程，以及计算流体等领域。

### 1.2 功能目标

> 在本任务中，作者根据[NSFnets (Navier-Stokes Flow nets): Physics-informed
neural networks for the incompressible Navier-Stokes
equations](https://arxiv.org/abs/2003.06496)中的代码[NSFNet code](https://github.com/Alexzihaohu/NSFnets/blob/master/)
>
> 复现Turbulent channel flow 的案例和指标。

### 1.3 意义

> 基于以paddle为backend的[DeepXDE](https://deepxde.readthedocs.io/en/latest/index.html)框架实现纳韦斯托克方程问题中的各项同性瞬态湍流流
场的高精度数据集[JHTDB](https://turbulence.pha.jhu.edu/)的复现。

## 2. PaddleScience 现状

> PaddleScience 暂时无法完成训练，从[AIstudio](https://aistudio.baidu.com/studio/project/partial/verify/6832363/fa46b783a28442b88fb7d2756ffddb6c)中的后台任务_nfs1_可以看到，速度的相对误差在O(10^(-2))附近，因此转而尝试使用基于paddle框架的DeepXDE。

## 3. 目标调研

>论文对纳韦斯托克方程进行了高精度求解，其难点主要在于代码较为陈旧，为基于原PINN而编写的tensorflow v1.0 版本代码，同时训练数据集精度高(JHTDB)以及训练网络庞大，单核GPU难以完成相关训练。
>多核并行运算，从NSFnets论文中

"We use 20,000 points inside the domain, 6,644 points on the boundary sampled at each time step, together with 33,524 points at the initial time step to compute the loss function. We set the total number of iterations nit = 150 in one training epoch. There are 10 hidden layers in the VP-NSFnet with 300 neurons per layer."

>可以发现复现论文数据的显存要求要远远高于其在github上的代码中的参数，因此我们要使用[并行训练指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/cluster_quick_start_collective_cn.html)进行训练。


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
> **JHTDB**

| alpha=1 size 4*50 | paper  | code(without BFGS) | paddle(DeepXDE)  |
|-------------------|--------|--------------------|---------|
| u                 | / | /            | /  |
| v                 | /| /             |/ |
| p                 | / | /                  | / |

## 6. 可行性分析和排期规划

>202309 :  调研

>202310 ：基于TF以及DeepXDE的复现

>202311 ：基于PaddleScience的复现

>202312 ：整理项目产出，撰写案例文档

## 7. 影响面

> 帮助PaddleScience实现纳韦斯托克方程的高精度复现

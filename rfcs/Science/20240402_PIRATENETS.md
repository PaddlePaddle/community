# [ PIRATENETS 代码复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/【Hackathon 6th】开源贡献个人挑战赛科学计算任务合集.md#no41-piratenets-代码复现)

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | Pesuking |
| 提交时间      |   2024-4-1   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop 2.52 版本    |
| 文件名        | rfc_PIRATENETS_code_reproduction.md |

## 1. 概述

### 1.1 相关背景

> 题目： [NO.41 PIRATENETS论文复现](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no41-piratenets-%E4%BB%A3%E7%A0%81%E5%A4%8D%E7%8E%B0)
>
> 复现论文链接：https://arxiv.org/pdf/2402.00326.pdf
>
> 论文摘要：尽管物理信息神经网络（PINNs）已成为处理由偏微分方程（PDEs）控制的前向和逆向问题的流行深度学习框架，但众所周知，当采用更大更深的神经网络架构时，它们的性能会下降。论文的研究发现，这种违反直觉的行为根源在于使用了不适当初始化方案的多层感知器（MLP）架构，这导致了网络导数的可训练性差，并最终导致PDE残差损失的不稳定最小化。为了解决这个问题，论文引入了物理信息残差自适应网络（PirateNets），这是一种旨在促进深层PINN模型稳定高效训练的新型架构。PirateNets利用一种新颖的自适应残差连接，允许网络以浅层网络的形式初始化，并在训练过程中逐渐加深。论文还展示了所提出的初始化方案能够让论文将适当的归纳偏差编码到与给定PDE系统相对应的网络架构中。论文提供了全面的实证证据，表明PirateNets更容易优化，并且可以通过显著增加深度来提高精度，最终在各种基准测试中实现了最先进的结果。

### 1.2 功能目标

> 根据论文结构图与损失函数公式、给出的训练配置，使用 PaddleScience API 实现 PIRATENETS 模型结构并在 2+案例上复现精度，代码合并至 examples。

### 1.3 意义

> 复现PIRATENETS 模型代码，一个用于高效训练深层物理信息神经网络（PINNs）模型的新型骨干网络。

## 2. PaddleScience 现状

> PaddleScience 套件暂无 PirateNets代码案例，但是可以基于PaddleScience API实现该模型。

## 3. 目标调研

> - 论文解决的问题：在深度物理信息神经网络（PINNs）中，随着网络深度和规模的增加，模型预测误差和训练困难的问题。
>
> - 论文所提出的方法：尽管近年来取得了显著进展，但大多数现有的物理信息神经网络（PINNs）研究倾向于使用小型和浅层的网络架构，这使得深层网络的巨大潜力尚未被充分利用。为了弥补这一差距，论文提出了一种新的架构类别，名为物理信息残差自适应网络（PirateNets）。论文的主要贡献总结如下：
>   - 论文认为，PINNs最小化PDE残差的能力由网络导数的能力决定。
>   - 论文通过证明，对于二阶线性椭圆和抛物线PDEs，训练误差的收敛将导致解及其导数的收敛，来支持这一论点。
>   - 论文从理论和实证两方面揭示，常规的初始化方案（例如Glorot或He初始化）会导致MLP导数的问题性初始化，从而导致较差的可训练性。
>   - 论文引入PirateNets来解决病态初始化的问题，使得PINNs能够稳定和高效地扩展到利用深层网络。PirateNets中提出的物理信息初始化也作为一种新方法，在模型初始化阶段整合物理先验知识。
>   - 论文进行了全面的数值实验，证明PirateNets在各种基准测试中实现了一致的准确性、鲁棒性和可扩展性的改进。
>
> ![image-20240401155420409](.\images\20240402_piratenets\piratenets.png)
>
> - 论文复现目标：实现 PIRATENETS 模型结构并在 2+案例上复现精度.
>
> ![image-20240401155407070](.\images\20240402_piratenets\benchmark.png)
>
> - 存在的难点：模型代码暂未开源，需要对论文有深入的理解。

## 4. 设计思路与实现方案

> 设计思路：参考 [PaddleScience 文档](https://paddlescience-docs.readthedocs.io/zh/latest/)对PirateNets模型进行复现，利用论文作者提供的数据进行模型训练，在论文中的2+案例（Allen-Cahn、Grey-Scott等）上复现精度。
>
> 实现方案：
>
> - 结合[论文](https://arxiv.org/pdf/2402.00326.pdf)提供的模型结构以及 [PaddleScience 文档](https://paddlescience-docs.readthedocs.io/zh/latest/)，利用PaddleScience API实现PirateNets模型。
>
>   - 利用嵌入函数$\Phi(x)$使用随机傅里叶特征（Random Fourier Features）将输入坐标映射到高维特征空间，其中$B \in R^{m \times d}$，$B\left[ i \right]\left[ j \right] \in \mathcal{N}\left(0, s^2\right)$，$s>0$。$B$中每个元素都是独立同分布的，$s$为用户指定的超参数。
>
>   $$
>   \Phi(x)=\left[\begin{array}{c}
>     \cos (B \mathrm{x}) \\
>     \sin (\mathrm{Bx})
>     \end{array}\right]
>   $$
>
>   - 将$\Phi(x)$输入两个全连接层，得到$U$、$V$，将其作为每个残差块的门控单元。
>     $$
>     \mathbf{U}=\sigma\left(\mathbf{W}_1 \Phi(\mathbf{x})+\mathbf{b}_1\right), \quad \mathbf{V}=\sigma\left(\mathbf{W}_2 \Phi(\mathbf{x})+\mathbf{b}_2\right),
>     $$
>
>    - 根据以下公式构建PirateNets模型残差块，其中$\odot$表示点乘，$\alpha^{(l)} \in \mathbb{R}$为可训练的参数。所有权重都使用Glorot方案进行初始化，而偏置则初始化为零。
>      $$
>      \begin{aligned}
>      \mathbf{f}^{(l)} & =\sigma\left(\mathbf{W}_1^{(l)} \mathbf{x}^{(l)}+\mathbf{b}_1^{(l)}\right), \\
>      \mathbf{z}_1^{(l)} & =\mathbf{f}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{f}^{(l)}\right) \odot \mathbf{V}, \\
>      \mathbf{g}^{(l)} & =\sigma\left(\mathbf{W}_2^{(l)} \mathbf{z}_1^{(l)}+\mathbf{b}_2^{(l)}\right) \\
>      \mathbf{z}_2^{(l)} & =\mathbf{g}^{(l)} \odot \mathbf{U}+\left(1-\mathbf{g}^{(l)}\right) \odot \mathbf{V}, \\
>      \mathbf{h}^{(l)} & =\sigma\left(\mathbf{W}_3^{(l)} \mathbf{z}_2^{(l)}+\mathbf{b}_3^{(l)}\right) \\
>      \mathbf{x}^{(l+1)} & =\alpha^{(l)} \mathbf{h}^{(l)}+\left(1-\alpha^{(l)}\right) \mathbf{x}^{(l)},
>      \end{aligned}
>      $$
>
>   - 模型最终的输出。
>     $$
>     \mathbf{u}_\theta=\mathbf{W}^{(L+1)} \mathbf{x}^{(L)}
>     $$
>   
> - 利用最小二乘法将各种类型的现有数据集成到网络的初始化阶段，对模型最后的线性层的参数进行初始化。
>   $$
>   \min _{\mathbf{W}}\|\mathbf{W} \Phi-\mathbf{Y}\|_2^2 .
>   $$
>
> - 利用作者提供的数据对模型进行训练以及评估。

### 4.1 补充说明[可选]

> 关于PINN的介绍：在机器学习中嵌入物理原则的最受欢迎和最灵活的方法之一是通过定制损失函数的形式。这些损失函数作为软约束，使机器学习模型在训练过程中偏向于尊重潜在的物理规律，从而促成了物理信息神经网络（PINNs）的出现。得益于它们的灵活性和易于实施的特点，PINNs已被广泛用于解决涉及PDEs的前向和逆向问题，通过无缝集成嘈杂的实验数据和物理定律到学习过程中。近年来，PINNs在计算科学的各个领域取得了一系列有希望的成果，包括在流体力学、生物工程和材料科学中的应用。此外，PINNs在分子动力学、电磁学、地球科学以及在设计热系统中也被有效应用。尽管近期的研究展示了PINNs的一些经验成功，但它们也突显出几个训练病理问题。这些问题包括频谱偏差、反向传播梯度不平衡和因果关系违反，所有这些都代表着研究和方法学发展的开放领域。为了解决这些问题，许多研究集中在通过改进神经网络架构和训练算法来提升PINNs的性能。值得注意的努力包括损失权重方案和关键点的自适应重采样，例如重要性采样、进化采样和基于残差的自适应采样。同时，在开发新的神经网络架构以改进PINNs的表征能力方面也取得了重大进展，包括自适应激活函数、位置嵌入和创新架构。进一步的探索包括替代性的目标函数，例如那些采用数值微分技术和受有限元方法（FEM）启发的变分形式，以及额外的正则化项以加速PINNs收敛。训练策略的发展也是一个活跃的研究领域，其中序列训练和迁移学习在加速学习和提高准确性方面显示出了希望。

## 5. 测试和验收的考量

> 成功复现PirateNets模型，并在论文中的2+案例（Allen-Cahn、Grey-Scott等）上复现精度。

## 6. 可行性分析和排期规划

> 2024.04.1 ~ 2024.05.1：精度论文，调研以及学习通过PaddleScience API构建模型的过程
>
> 2024.05.1 ~ 2024.06.1：复现模型，训练以及评估，完成相应文档
>
> 2024.06.1 ~ 2024.06.12：完善优化代码、文档，提交PR
>
> 

## 7. 影响面

> 丰富[PaddleScience](https://paddlescience-docs.readthedocs.io/zh/latest/)的应用案例，在example目录下增加PirateNets模型。  

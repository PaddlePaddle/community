# 为 PaddleScience 2D 非定常圆柱绕流案例进行训练过程动力学行为进行分析

|                                                                |                                                         |
| -------------------------------------------------------------- | ------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | PuQing                                                  |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-07-08                                              |
| 版本号                                                         | V1.0                                                    |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                                 |
| 文件名                                                         | 20220706_analysis_of_2d_cylinder_flow_for_paddlescience |

## 一、概述

PaddleScience 工具组件已有的 2D 非定常圆柱绕流 Demo ，该任务需要在此基础上进行训练过程动力学行为分析，比如时间维度、空间维度、频域维度收敛情况。

## 二、设计思路与实现方案

### 时间维度

参考论文[LEARNING CONTINUOUS-TIME PDES FROM SPARSE DATA WITH GRAPH NEURAL NETWORKS](https://arxiv.org/abs/2006.08956)中图 2。
可以画出网络误差随着时间上的变化，以及网络的训练过程中的收敛情况，具体为：

对于网络输出$u,v,p=f(t,x,y)$在一个训练周期上的相对误差可以计算为

$$
\delta(t)=\left ( \sum_x\sum_y \frac{|f-y|}{|y|}  \right )
$$

这样可以画出时间域上的误差随着训练过程的收敛情况。可以做出$\delta(t,epoch)$图

### 空间维度

同样的，对于空间维度，可以对时间上的误差进行求和，然后画出空间域上的误差随着训练过程的收敛情况。即：

$$
\delta(x,y)=\left ( \sum_t \frac{|f-y|}{|y|}  \right )
$$

### 频率维度

参考于论文[FREQUENCY PRINCIPLE: FOURIER ANALYSIS SHEDS LIGHT ON DEEP NEURAL NETWORKS, 2020](https://arxiv.org/pdf/1901.06523.pdf)

> 此处讨论的频率为响应频率(response frequency)，对于 2D 非定常圆柱绕流网络来说映射函数$f(x)$为：
>
> $$
> f(t,x,y):\mathbb{R}^3\to \mathbb{R}^3
> $$

由于网络输出为三个维度，即$\hat{u},\hat{v},\hat{p}$,，分别为垂直速度，水平速度，压强大小。所以分别讨论$\hat{u},\hat{v},\hat{p}$三个维度的情况。以$\hat{u}$为例，对映射函数$u(t,x,y)$做傅里叶变换，有：

$$
\mathcal{F}\hat{u}(\vec{\xi })=\int_{\mathbb{R}^n}e^{-2\pi \mathrm{i}(\vec{x}\cdot \vec{\xi })}\hat{u}(\vec{x})d\vec{x}\tag{矢量式}
$$

其中$\vec{\xi}$的维度与$\vec{x}$相同，对于像$(t,x,y)$高维度张量很难进行频率分析，所以一般有如下两个方法：

- 投影法

  选取一个方向向量$\vec{p}$，可以将$\vec{x}$沿着方向向量$\vec{p}$进行投影，使得$\vec{v}=\vec{p}\cdot\vec{x}\in \mathbb{R}$，这样便可以对降维后的向量$\vec{v}$进行傅里叶变换，得到$\mathcal{F}\hat{u}(\xi)$，其中的$\xi$是一维张量，这样便可方便对频率上的误差分析，给定 GT 上的函数值$u(t,x,y)$，则系统频率相对误差为

  $$
   \delta =\frac{\mathcal{F}[\hat{u} ]-\mathcal{F}[u]}{\mathcal{F}[u]}
  $$

  对于降维可以对输出做 PCA 取出第一主成分，再进行 DFT 变换，计算出误差

- Filtering 方法

  在频率空间乘以一个示性函数，例如二维傅里叶变换，$\mathcal{F}[g(x,y)](u,v)$将空间域转换到频率域，示性函数可以如下图构造：

  $$
  \mathbb{l} _{|\xi|\le \xi_0}=\left\{\begin{matrix}
  1, & |\xi|\le \xi_0,\\
  0, & |\xi|>\xi_0.
  \end{matrix}\right.
  $$

  ![dd8a632ae3d0ce0386d66d737c94927c96caa8bf136b9c11.png](https://images.puqing.work/dd8a632ae3d0ce0386d66d737c94927c96caa8bf136b9c11.png)

  当频率中向量$|\xi|$小于等于$\xi_0$时可以认为是低频，当大于时可以认定为是高频，但是高纬度的傅里叶变换的计算成本很高，所以一般对输出量做高斯低通滤波以及高斯高通滤波，以减少高纬度的计算成本。所以频率误差可以写为：

  $$
    e_{\text {low }}=\left(\frac{\sum_{i}\left|\mathbf{h}_{i}^{\text {low }, \delta}-\mathbf{\hat{h}}_{i}^{\text {low }, \delta}\right|^{2}}{\sum_{i}\left|\mathbf{h}_{i}^{\text {low }, \delta}\right|^{2}}\right)^{\frac{1}{2}}, \quad e_{\text {high }}=\left(\frac{\sum_{i}\left|\mathbf{h}_{i}^{\text {high }, \delta}-\mathbf{\hat{h} }_{i}^{\mathrm{high}, \delta}\right|^{2}}{\sum_{i}\left|\mathbf{h}_{i}^{\mathrm{high}, \delta}\right|^{2}}\right)^{\frac{1}{2}}
  $$

  本提案将会尝试上述两种方法，分析网络在频率误差上随训练过程的变化。

## 三、测试和验收的考量

上述分析应写成 API 的形式，通过在指定位置收集训练过程数据，在训练完成后给出上述分析图表。

图表要求：应有说明性文字，以及图例

## 四、可行性分析和排期规划

可在当前任务时间内完成

## 参考文献及代码

[LEARNING CONTINUOUS-TIME PDES FROM SPARSE DATA WITH GRAPH NEURAL NETWORKS](https://arxiv.org/abs/2006.08956)

[FREQUENCY PRINCIPLE: FOURIER ANALYSIS SHEDS LIGHT ON DEEP NEURAL NETWORKS, 2020](https://arxiv.org/pdf/1901.06523.pdf)

[OPEN Learning aerodynamics with neural network](https://www.researchgate.net/publication/360191315_Learning_aerodynamics_with_neural_network)

[F-Principle](https://github.com/Mark-Sky/Fourier-and-AI/tree/main/F-Principle)

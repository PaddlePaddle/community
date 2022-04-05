# 增加paddle作为DeepXDE的 backend

| |  |
|---|---|
|提交作者 | 梁嘉铭 |
|提交时间 | 2022-04-04 |
|版本号 | V1.2 |
|依赖飞桨版本 | develop版本 |
|文件名 | 20220310_DeepXDE_design_for_paddle.md|

# 一、概述

## 1、相关背景

[lululxvi/deepxde/issues/559](https://github.com/lululxvi/deepxde/issues/559)

## 2、功能目标

1. Support function approximation
   
   Goal: Support the following test examples: func.py, dataset.py

2. Support solving forward ODEs
   
   Goal: Support the following test examples: A simple ODE system

3. Support solving inverse ODEs

    Goal: Support the following test examples: Inverse problem for the Lorenz system, Inverse problem for the Lorenz system with exogenous input


## 3、意义

[Deepxde](https://github.com/lululxvi/deepxde)是一个非常有用的求解函数逼近，正向/逆常/偏微分方程 (ODE/PDE)代码仓库。该功能会增加Paddle在科学计算方面的使用。

# 二、飞桨现状

1. 例如`paddle.sin()`,`paddle.cos()`算子的求高阶导数，所以不支持`apply_feature_transform()`以及`apply_output_transform()`函数。
考虑参考[#32188](https://github.com/PaddlePaddle/Paddle/pull/32188)pr，对sin/cos函数的求二阶导数算子进行支持。具体将在[#L93](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/funcs/activation_functor.h#L93)行添加`SinGradGradFunctor`方法，以及`CosGradGradFunctor`方法，并在[L267](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/activation_op.h#L267)处进行调用，在[L1477](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/activation_op.cc#L1477)处进行注册。

1. Paddle近期合入了`L-BFGS`优化器以及`BFGS`优化器，调用方式为函数式调用

# 三、业内方案调研

Tensorflow、Pytorch、jax均在deepxde中得到支持。

## L-BFGS优化器

其中Pytorch的[`L-BFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS)优化器可以类似于optimizer式的调用，而[tensorflow](https://tensorflow.google.cn/probability/api_docs/python/tfp/optimizer/lbfgs_minimize?hl=zh-cn)中是类似于Paddle合入的`L-BFGS`优化器，通过构造一个待优化的函数进行优化。


# 四、设计思路与实现方案

## 命名与参数设计

在代码中PaddlePaddle的名称会被书写为paddle。

## API实现方案

实现方式主要参考`Pytorch`的实现方式。

## 求解Jacobian

在paddle中已经通过[commit提交](https://github.com/PaddlePaddle/Paddle/commit/ec2f68e85d413655d5774d03fb81c5ba13db54cd)对Jacobian进行了支持，该求导方式就是通过`paddle.grad()`对函数求导，而deepxde已经实现了Jacobian类，所以只需要求解y对x的导即可，使用`paddle.autograd.grad`进行求导，具体为

```python
self.J[i] = paddle.autograd.grad(y, self.xs, create_graph=True, retain_graph=True)[0]
```

含义为求解y对x的导，并且保留求导过程中的计算图，以便后续求解y对x的二阶导。

## 静态图

与类似于动态图的构建方式一样，通过

# 六、测试和验收的考量

根据任务要求，测试在excample中如下文件。
-  func.py, dataset.py
-  ode_system.py
-  Lorenz_inverse.py, Lorenz_inverse_forced.ipynb

# 七、可行性分析和排期规划

paddle日益完善，基础算子以及函数均有支持，但是暂时不支持L-BFGS方法，以及对高阶导数。总体可以在活动时间内完成。

# 八、影响面
对deepxde新增backend, 无影响

# 九、Feature
pinn中需要利用到sin/cos的高阶导数，所以希望未来Paddle可以支持sin/cos的高阶导数，同时为了支持L-BFGS方法，需要添加相关优化器的实现，以便优化loss。
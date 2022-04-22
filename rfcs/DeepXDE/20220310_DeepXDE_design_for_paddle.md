# 增加 paddle 作为 DeepXDE 的 backend

|              |                                       |
| ------------ | ------------------------------------- |
| 提交作者     | 梁嘉铭                                |
| 提交时间     | 2022-04-22                            |
| 版本号       | V1.3                                  |
| 依赖飞桨版本 | develop 版本                          |
| 文件名       | 20220310_DeepXDE_design_for_paddle.md |

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

[Deepxde](https://github.com/lululxvi/deepxde)是一个非常有用的求解函数逼近，正向/逆常/偏微分方程 (ODE/PDE)代码仓库。该功能会增加 Paddle 在科学计算方面的使用。

# 二、飞桨现状

1. 例如`paddle.sin()`,`paddle.cos()`算子的求高阶导数，所以不支持`apply_feature_transform()`以及`apply_output_transform()`函数。
   考虑参考[#32188](https://github.com/PaddlePaddle/Paddle/pull/32188)pr，对 sin/cos 函数的求二阶导数算子进行支持。具体将在[#L93](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/funcs/activation_functor.h#L93)行添加`SinGradGradFunctor`方法，以及`CosGradGradFunctor`方法，并在[L267](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/activation_op.h#L267)处进行调用，在[L1477](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/activation_op.cc#L1477)处进行注册。

1. Paddle 近期合入了`L-BFGS`优化器以及`BFGS`优化器，调用方式为函数式调用

# 三、业内方案调研

Tensorflow、Pytorch、jax 均在 deepxde 中得到支持。

## L-BFGS 优化器

其中 Pytorch 的[`L-BFGS`](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS)优化器可以类似于 optimizer 式的调用，而[tensorflow](https://tensorflow.google.cn/probability/api_docs/python/tfp/optimizer/lbfgs_minimize?hl=zh-cn)中是类似于 Paddle 合入的`L-BFGS`优化器，通过构造一个待优化的函数进行优化。

但是考虑到 L-BFGS 优化器输入为一维向量，所以在输入优化的权重参数时，需要将其转换为一维向量。可参考 TensorFlow 的实现，[tfp_optimizer](https://github.com/lululxvi/deepxde/blob/master/deepxde/optimizers/tensorflow/tfp_optimizer.py)。

# 四、设计思路与实现方案

## 命名与参数设计

在代码中 PaddlePaddle 的名称会被书写为 paddle。

## API 实现方案

实现方式主要参考`Pytorch`的实现方式。

## 求解 Jacobian

在 paddle 中已经通过[commit 提交](https://github.com/PaddlePaddle/Paddle/commit/ec2f68e85d413655d5774d03fb81c5ba13db54cd)对 Jacobian 进行了支持，该求导方式就是通过`paddle.grad()`对函数求导，而 deepxde 已经实现了 Jacobian 类，所以只需要求解 y 对 x 的导即可，使用`paddle.autograd.grad`进行求导，具体为

```python
self.J[i] = paddle.autograd.grad(y, self.xs, create_graph=True, retain_graph=True)[0]
```

含义为求解 y 对 x 的导，并且保留求导过程中的计算图，以便后续求解 y 对 x 的二阶导。

## 静态图

为提前加入对 sin/cos 等算子的高阶导数，我们将模型默认设置为静态图模式，具体是在模型 forward()函数前用` @paddle.jit.to_static`进行自动转换。

# 五、测试和验收的考量

根据任务要求，测试在 excample 中如下文件。

- func.py, dataset.py
- ode_system.py
- Lorenz_inverse.py, Lorenz_inverse_forced.ipynb

# 六、可行性分析和排期规划

实现功能已经完成。总体可以在活动时间内完成。

# 七、影响面

对 deepxde 新增 backend, 无影响。

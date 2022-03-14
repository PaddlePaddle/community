#paddle.optimizer.lr.OneCycleLR设计文档

| API名称                                    | paddle.optimizer.lr.OneCycleLR           |
| ---------------------------------------- | ---------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden"> | Asthestarsfalll                          |
| 提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-12                               |
| 版本号                                      | V1.0                                     |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0                                   |
| 文件名                                      | 20210312_api_design_for_one_cycle_lr.md<br> |

# 一、概述

## 1、相关背景

 OneCycleLR最早在[Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)中提出，它在整个训练过程中只有一个周期（OneCycle），即从小学习率上升至最大学习率，然后又下降至低于初始学习率的大小，它的一个特征是每个batch都进行学习率调整。

## 2、功能目标

在 Paddle 框架中，新增 OneCycleLR 优化调度器，调用路径为：paddle.optimizer.lr.OneCycleLR。

## 3、意义

飞桨支持OneCycleLR 优化调度器

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=- 1, verbose=False)`

在pytorch中，介绍为：

```
Sets the learning rate of each parameter group according to the 1cycle learning rate policy. The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate. This policy was initially described in the paper Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.

The 1cycle learning rate policy changes the learning rate after every batch. step should be called after a batch has been used for training.

This scheduler is not chainable.

Note also that the total number of steps in the cycle can be determined in one of two ways (listed in order of precedence):

A value for total_steps is explicitly provided.

A number of epochs (epochs) and a number of steps per epoch (steps_per_epoch) are provided. In this case, the number of total steps is inferred by total_steps = epochs * steps_per_epoch

You must either provide a value for total_steps or provide a value for both epochs and steps_per_epoch.

The default behaviour of this scheduler follows the fastai implementation of 1cycle, which claims that “unpublished work has shown even better results by using only two phases”. To mimic the behaviour of the original paper instead, set three_phase=True.
```

### 实现方法

在实现方法上, Pytorch直接使用python实现，[代码位置](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py#L1341)。

由于Paddle的lr_scheduler调用方式与Pytorch区别较大，因此做了修改，修改如下：

- 取消momentum的相关参数，因为不可能通过`lr_scheduler`修改`momentum`或者`beta`
- `max_lr`参数类型仅为`float`， 而非Pytorch中的`float or list`，当`max_lr`为`list` 时，Pytorch会为`optimizer.param_groups` 内的各个元素分配不同的学习率，而paddle的`lr_scheduler`并不能获取到`optimizer`，因此可由用户在外部自行实现。

实现方法与Pytorch基本相同。

# 四、对比分析

# 五、方案设计

## 命名与参数设计

API设计为`paddle.optimizer.lr.OneCycleLR(max_lr, total_step=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', divide_factor=25., final_divide_factor=1e4, three_phase=False, last_epoch=-1, verbose=False)`

形参名`div_factor`->`divide_factor`, `final_div_factor`->`final_divide_factor`

## 底层OP设计

仅使用python实现

# 六、测试和验收的考量

测试考虑的case如下：

-  `paddle.optimizer.lr.OneCycleLR`和`torch.optim.lr_scheduler.OneCycleLR` 与`numpy`结果的数值一致性
- 错误检查：`total_step`, `epochs`和`steps_per_epoch`都未指定时能正确抛出错误，并且其数值小于等于0时能正确抛出错误；
- 错误检查：`anneal_strategy`不在指定范围时能正确抛出错误；
- 错误检查：`pct_start`值不在[0，1]时能正确抛出错误；

# 七、可行性分析及规划排期

方案直接依赖python，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 九、评审意见

（由评审人填写，开发者无需填写）

| 状态   | 提出人          | 问题   |
| ---- | ------------ | ---- |
| 通过   | dingjiaweiww | 无    |

# 名词解释

无

# 附件及参考资料

无
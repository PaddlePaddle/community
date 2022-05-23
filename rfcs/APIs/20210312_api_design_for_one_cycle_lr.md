#paddle.optimizer.lr.OneCycleLR设计文档

| API名称                                                      | paddle.optimizer.lr.OneCycleLR              |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                             |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-12                                  |
| 版本号                                                       | V1.1                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                     |
| 文件名                                                       | 20210312_api_design_for_one_cycle_lr.md<br> |

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

核心代码如下：

```python
def get_lr(self):
    if not self._get_lr_called_within_step:
        warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

    lrs = []
    step_num = self.last_epoch

    if step_num > self.total_steps:
        raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                            .format(step_num + 1, self.total_steps))

    for group in self.optimizer.param_groups:
        start_step = 0
        for i, phase in enumerate(self._schedule_phases):
            end_step = phase['end_step']
            if step_num <= end_step or i == len(self._schedule_phases) - 1:
                pct = (step_num - start_step) / (end_step - start_step)
                computed_lr = self.anneal_func(group[phase['start_lr']], group[phase['end_lr']], pct)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group[phase['start_momentum']], group[phase['end_momentum']], pct)
                break
            start_step = phase['end_step']

        lrs.append(computed_lr)
        if self.cycle_momentum:
            if self.use_beta1:
                _, beta2 = group['betas']
                group['betas'] = (computed_momentum, beta2)
            else:
                group['momentum'] = computed_momentum

    return lrs
```

## 其他实现

该[仓库](https://github.com/dkumazaw/onecyclelr/blob/master/onecyclelr.py)有一个基于Pytorch的实现，[代码位置](https://github.com/dkumazaw/onecyclelr/blob/master/onecyclelr.py#L4)

介绍为：

```
 Sets the learing rate of each parameter group by the one cycle learning rate policy
 proposed in https://arxiv.org/pdf/1708.07120.pdf. 
 It is recommended that you set the max_lr to be the learning rate that achieves 
 the lowest loss in the learning rate range test, and set min_lr to be 1/10 th of max_lr.
 So, the learning rate changes like min_lr -> max_lr -> min_lr -> final_lr, 
 where final_lr = min_lr * reduce_factor.
 Note: Currently only supports one parameter group.
```

核心代码：

```python
    def step(self):
        """Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum
```



# 四、对比分析

两种实现基于Pytorch，思路基本相同，Pytorch官方实现更加简洁完善，因此参考Pytorch代码进行实现。



# 五、方案设计

## 命名与参数设计

API设计为`paddle.optimizer.lr.OneCycleLR(max_learning_rate, total_steps, divide_factor=25., end_learning_rate=0.0001, phase_pct=0.3, anneal_strategy='cos', three_phase=False, last_epoch=-1, verbose=False)`

形参名`div_factor`->`divide_factor`, `max_lr`->`max_learning_rate`。

参数：

- max_learning_rate：训练过程中的最大学习率；
- total_steps：训练总步数；
- divide_factor：初始学习率`initial_learning_rate`由`max_learning_rate`/`divide_factor`确定，默认为25；
- end_learning_rate：训练过程中的最小学习率，应该是一个远小于初始学习率的数。
- phase_pct：提高学习率所需步数占总训练步数的比例，默认：0.3；
- anneal_strategy：学习率退火策略，'cos'或'linear'，默认为'cos'；
- three_phase：若为`True`，则使用三阶段的调度策略，即学习率先由`initial_learning_rate`上升至`max_learning_rate`，再下降回`initial_learning_rate`，最后下降到`min_learning_rate`；若为False，则使用两阶段的调度策略，即学习率先由`initial_learning_rate`上升至`max_learning_rate`，再直接下降到`min_learning_rate`。默认为`False`；
- last_epoch：可选，上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为 -1，则为初始学习率；
- verbose：可选，如果是 `True` ，则在每一轮更新时在标准输出 stdout 输出一条信息。默认值为 `False` 。


## 底层OP设计

仅使用python实现，无需设计底层OP。

## API实现方案

由于Paddle的lr_scheduler调用方式与Pytorch区别较大，因此做了修改，修改如下：

- 取消momentum的相关参数，因为不可能通过`lr_scheduler`修改`momentum`或者`beta`
- `max_lr`参数类型仅为`float`， 而非Pytorch中的`float or list`，当`max_lr`为`list` 时，Pytorch会为`optimizer.param_groups` 内的各个元素分配不同的学习率，而paddle的`lr_scheduler`并不能获取到`optimizer`，因此可由用户在外部自行实现。

实现方法与Pytorch基本相同。

# 六、测试和验收的考量

测试考虑的case如下：

-  `paddle.optimizer.lr.OneCycleLR`与`numpy`结果的数值一致性
- 错误检查：`total_step`, `epochs`和`steps_per_epoch`都未指定时能正确抛出错误，并且其数值小于等于0时能正确抛出错误；
- 错误检查：`anneal_strategy`不在指定范围时能正确抛出错误；
- 错误检查：`pct_start`值不在[0，1]时能正确抛出错误；

# 七、可行性分析及规划排期

方案直接依赖python，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
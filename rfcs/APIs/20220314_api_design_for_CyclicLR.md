# paddle.optimizer.lr.CyclicLR设计文档

| API名称                                                      | paddle.optimizer.lr.CyclicLR        |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                     |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-14                          |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0                              |
| 文件名                                                       | 20220314_design_for_CyclicLR.md<br> |

# 一、概述

## 1、相关背景

CyclicLR最早在论文[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)中提出，它控制学习率以固定周期在base_lr和max_lr之间变化，它的调整是以batch.step进行的，同时，它还有3种内置的波幅调整策略：“triangular”、“triangular2”、“exp_range”。

## 2、功能目标

在 Paddle 框架中，新增 CyclicLR 优化调度器，调用路径为：[paddle.optimizer.lr](http://paddle.optimizer.lr/).CyclicLR。

## 3、意义
飞桨支持CyclicLR优化调度器

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API``，以及对应的`torch.optim.lr_scheduler.``CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,mode='triangular',gamma=1.,scale_fn=None,scale_mode='cycle',cycle_momentum=True,base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)`.

在pytorch中，介绍为：

```
Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR). The policy cycles the learning rate between two boundaries with a constant frequency, as detailed in the paper Cyclical Learning Rates for Training Neural Networks. The distance between the two boundaries can be scaled on a per-iteration or per-cycle basis.

Cyclical learning rate policy changes the learning rate after every batch. step should be called after a batch has been used for training.

This class has three built-in policies, as put forth in the paper:

“triangular”: A basic triangular cycle without amplitude scaling.

“triangular2”: A basic triangular cycle that scales initial amplitude by half each cycle.

“exp_range”: A cycle that scales initial amplitude by \text{gamma}^{\text{cycle iterations}}gamma 
cycle iterations at each cycle iteration.

This implementation was adapted from the github repo: bckenstler/CLR
```

### 实现方法

在实现方法上, Pytorch是直接通过python实现的，[代码位置](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py#L994)。

核心代码为：

```python
def get_lr(self):
    """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

    if not self._get_lr_called_within_step:
        warnings.warn("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
			scale_factor = (x - 1) / (self.step_ratio - 1)

		lrs = []
		for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
			base_height = (max_lr - base_lr) * scale_factor
			if self.scale_mode == 'cycle':
				lr = base_lr + base_height * self.scale_fn(cycle)
			else:
				lr = base_lr + base_height * self.scale_fn(self.last_epoch)
				lrs.append(lr)

         if self.cycle_momentum:
         momentums = []
         for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
			base_height = (max_momentum - base_momentum) * scale_factor
			if self.scale_mode == 'cycle':
				momentum = max_momentum - base_height * self.scale_fn(cycle)
			else:
				momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
			momentums.append(momentum)
			for param_group, momentum in zip(self.optimizer.param_groups, momentums):
				param_group['momentum'] = momentum

		return lrs
```



## 其他实现

https://github.com/bckenstler/CLR

该仓库使用tensorflow2进行了实现

介绍为：

```
A cyclical learning rate is a policy of learning rate adjustment that increases the learning rate off a base value in a cyclical nature. Typically the frequency of the cycle is constant, but the amplitude is often scaled dynamically at either each cycle or each mini-batch iteration.
```



### 实现方法

直接使用python，[代码位置](https://github.com/bckenstler/CLR/blob/master/clr_callback.py#L5)

核心代码为：

```python
def clr(self):
	cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
	x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
	if self.scale_mode == 'cycle':
		return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
	else:
		return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
```

# 四、对比分析

Pytorch官方文档中提到参考了上述tensorflow仓库进行修改实现，Pytorch与上述tensorflow仓库的主要区别如下：

- 在该tensorflow仓库中，参数`step_size `表示从`base_lr`上升到`max_lr`（从`max_lr`下降到`base_lr`）的所需步数，而Pytorch使用两个参数`step_size_up`和`step_size_down`来分别表示学习率上升和下降所需的步数；
- 引入动量衰减策略。

其余实现基本相同，且Pytorch代码更易理解，故参考Pytorch进行实现。

# 五、方案设计

## 命名与参数设计

API设计为`paddle.optimizer.lr.CyclicLR(base_learning_rate,max_learning_rate,step_size_up,step_size_down, mode='triangular',exp_gamma=1.,scale_fn=None,scale_mode='cycle',last_epoch=-1,verbose=False)`

去除了Pytorch中`momentum`的相关参数。

同时，为了保持与paddle其他lrscheduler相关的api保持一致，将`base_lr`修改为`base_learning_rate`，`max_lr`修改为`max_learning_rate`。

## 底层OP设计

直接使用python实现，不再单独设计OP。

## API实现方案

主要参考Pytorch进行实现，实现位置为`paddle/optimizer/lr.py`。

# 六、测试和验收的考量

测试考虑的case如下：

- 动态图，静态图，与numpy的结果保持一致；
- 错误检查：`step_size_up`和 `step_size_down`数值小于等于0时能正确抛出错误；
- 错误检查：`scale_mode`不在可选范围内时能正确抛出异常；

# 七、可行性分析及规划排期

方案仅使用python实现，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

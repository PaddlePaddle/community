
# paddle.optimizer.lr.LinearLR设计文档

| API名称                                                      | paddle.optimizer.lr.LinearLR        |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                     |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-09-25                          |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                              |
| 文件名                                                       | 20230925_api_design_for_LinearLR.md<br> |

# 一、概述

## 1、相关背景


LinearLR 学习率调度器在训练开始时通过乘法因子降低学习率, 但是它会在一定数量的训练步骤中线性地改变学习率，直到它达到最终设定的学习率。

## 2、功能目标

在 Paddle 框架中，新增 LinearLR 优化调度器，调用路径为：paddle.optimizer.lr.LinearLR。

## 3、意义
飞桨支持LinearLR优化调度器

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API``，以及对应的`torch.optim.lr_scheduler.``LinearLR(optimizer, start_factor, end_factor, total_iters, last_epoch=-1, verbose=False)`.

在pytorch中，介绍为：

```
Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

```

### 实现方法

在实现方法上, Pytorch是直接通过python实现的，[代码位置](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L551)。

核心代码为：

```python
def get_lr(self):
    if not self._get_lr_called_within_step:
        warnings.warn("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", UserWarning)

    if self.last_epoch == 0:
        return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

    if self.last_epoch > self.total_iters:
        return [group['lr'] for group in self.optimizer.param_groups]

    return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
            (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
            for group in self.optimizer.param_groups]
```


# 四、方案设计

## 命名与参数设计

API设计为`paddle.optimizer.lr.LinearLR(base_learning_rate,total_steps,start_factor,end_factor,last_epoch=-1,verbose=False)`

将默认参数 `total_iters` 改为非默认参数 `total_steps`， 名称上与其他学习率调度器保持一致。


## 底层OP设计

直接使用python实现，不再单独设计OP。

## API实现方案

参考Pytorch进行实现，实现位置为`paddle/optimizer/lr.py`。

# 六、测试和验收的考量

测试考虑的case如下：

- 动态图，静态图，与numpy的结果保持一致；
- 错误检查：`start_factor`和 `end_factor`数值不正确时时能正确抛出错误；
- 错误检查：`total_steps` 小于等于0时抛出错误；

# 七、可行性分析及规划排期

方案仅使用python实现，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

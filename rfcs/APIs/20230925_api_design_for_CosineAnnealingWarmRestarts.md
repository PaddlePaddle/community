# paddle.CosineAnnealingWarmRestarts 设计文档

| API名称                                                      | paddle.optimizer.lr.CosineAnnealingWarmRestarts |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者   | NetPunk                   |
| 提交时间| 2023-09-25                 |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名                                                       | 20230925_api_design_for_CosineAnnealingWarmRestarts.md |

# 一、概述

## 1、相关背景

余弦退火学习率具有良好的1个核心思想，即高学习率时段和低学习率时段周期性出现。高学习率时段的功能是防止学习者陷入局部成本最小化；低学习率时段允许在（希望）找到的全局最小值内收敛到接近真实的最优点。具有热重启的余弦退火学习率是在常规余弦退火学习率算法基础上的一个改进，能够控制学习率的回升速度，这样到了训练后期，学习率回升次数变少或不会再回升，保持学习率一直下降直到训练结束。

## 2、功能目标

实现CosineAnnealingWarmRestarts余弦退火学习率，调用路径为：

- paddle.optimizer.lr.CosineAnnealingWarmRestarts

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有CosineAnnealingWarmRestarts API（https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html）

在 PyTorch 文档中，介绍为：

```
Set the learning rate of each parameter group using a cosine annealing
schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
is the number of epochs since the last restart and :math:`T_{i}` is the number
of epochs between two warm restarts in SGDR:

.. math::
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
\cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

It has been proposed in
`SGDR: Stochastic Gradient Descent with Warm Restarts`_.
```
输入优化器和参数`T_0`、`T_mult`、`eta_min`、`last_epoch`，即可得到学习率优化类

### 实现方法

PyTorch采用的是python端实现，封装为类

```python
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
```

去除掉过程规范化逻辑，其核心计算函数为`step`和`get_lr`

# 四、对比分析

可以直接参考的实现是pytorch，因为paddle和pytorch的学习率优化函数在行为上比较相似，因此大致逻辑可以套用pytorch实现

# 五、方案设计

## 命名与参数设计

API的实现为一个类，方法组成和lr中其它类相似

paddle.optimizer.lr.CosineAnnealingWarmRestarts
----------------------
参数
:::::::::

- `leaning_rates` - 初始化的学习率
- `T_0` - 第一次重启的迭代次数
- `T_mult` - 乘积因子，用于重启后增加`T_i`，默认值为1
- `eta_min` - 最小学习率，默认值为0
- `last_epoch` - 上一轮的轮数，重启训练时设置为上一轮的 epoch 数。默认值为 -1
- `verbose` -   如果是 `True`，则在每一轮更新时在标准输出 stdout 输出一条信息。默认值为 `False` 。



## 底层OP设计

python端API组合实现

## API实现方案

参考pytorch逻辑和lr中其它类的实现，可以得到初版API实现代码如下

~~~python
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, learning_rate, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2               

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
~~~



# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：实现测试方法`cosine_annealing_warm_restarts_lr`和测试类`TestCosineAnnealingWarmRestarts`，计算n个epoch后学习率的正确性
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：`T_0`和`T_mult`的值和范围，`epoch`范围。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

为独立新增API，对其他模块没有影响
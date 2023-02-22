# paddle.distribution.Geometric 设计文档

| API 名称     | paddle.distribution.Geometric        |
| ------------ | ---------------------------------- |
| 提交作者     | 李文博（李长安）倪浩（TUDelftHao） |
| 提交时间     | 2023-02-22                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.4.2                             |
| 文件名       | 20230220_design_for_Geometric_distribution.md     |

# 一、概述

## 1、相关背景

此任务的目标是在 Paddle 框架中，基于现有概率分布方案进行扩展，新增 Geometric API， API调用 `paddle.distribution.Geometric`。

## 2、功能目标

增加 API `paddle.distribution.Geometric`，Geometric 用于 Geometric 分布的概率统计与随机采样。API具体包含如下方法：

功能：`Creates a Geometric distribution parameterized by loc and scale.`

- `mean`计算均值；
- `mode` 众数；
- `variance`计算方差 ；
- `stddev`计算标准偏差
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `entropy` 熵计算；
- `cdf` 累积分布函数(Cumulative Distribution Function)
- `icdf` 逆累积分布函数
- `kl_divergence` 返回两个分布之间的kl散度

## 3、意义

为 Paddle 增加用于几何分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状 

- 目前 飞桨没有 API `paddle.distribution.Geometric`，

- 调研 Paddle 及业界实现惯例，并且代码风格及设计思路与已有概率分布保持一致代码，需采用飞桨2.0之后的API，故依赖飞桨版本V2.4.2 。

- PS：已经参与Laplace的API贡献，故此部分较为熟悉。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.Geometric.Geometric((probs=None, logits=None)`


### 源代码

```
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property
from torch.nn.functional import binary_cross_entropy_with_logits

__all__ = ['Geometric']

[docs]class Geometric(Distribution):
    r"""
    Creates a Geometric distribution parameterized by :attr:`probs`,
    where :attr:`probs` is the probability of success of Bernoulli trials.
    It represents the probability that in :math:`k + 1` Bernoulli trials, the
    first :math:`k` trials failed, before seeing a success.

    Samples are non-negative integers [0, :math:`\inf`).

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Geometric(torch.tensor([0.3]))
        >>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
        tensor([ 2.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`. Must be in range (0, 1]
        logits (Number, Tensor): the log-odds of sampling `1`.
    """
    arg_constraints = {'probs': constraints.unit_interval,
                       'logits': constraints.real}
    support = constraints.nonnegative_integer

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            self.probs, = broadcast_all(probs)
        else:
            self.logits, = broadcast_all(logits)
        probs_or_logits = probs if probs is not None else logits
        if isinstance(probs_or_logits, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = probs_or_logits.size()
        super(Geometric, self).__init__(batch_shape, validate_args=validate_args)
        if self._validate_args and probs is not None:
            # Add an extra check beyond unit_interval
            value = self.probs
            valid = value > 0
            if not valid.all():
                invalid_value = value.data[~valid]
                raise ValueError(
                    "Expected parameter probs "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to be positive but found invalid values:\n{invalid_value}"
                )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Geometric, _instance)
        batch_shape = torch.Size(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        super(Geometric, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    @property
    def mean(self):
        return 1. / self.probs - 1.

    @property
    def mode(self):
        return torch.zeros_like(self.probs)

    @property
    def variance(self):
        return (1. / self.probs - 1.) / self.probs

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        tiny = torch.finfo(self.probs.dtype).tiny
        with torch.no_grad():
            if torch._C._get_tracing_state():
                # [JIT WORKAROUND] lack of support for .uniform_()
                u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
                u = u.clamp(min=tiny)
            else:
                u = self.probs.new(shape).uniform_(tiny, 1)
            return (u.log() / (-self.probs).log1p()).floor()


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value, probs = broadcast_all(value, self.probs)
        probs = probs.clone(memory_format=torch.contiguous_format)
        probs[(probs == 1) & (value == 0)] = 0
        return value * (-probs).log1p() + self.probs.log()


    def entropy(self):
        return binary_cross_entropy_with_logits(self.logits, self.probs, reduction='none') / self.probs
```


# 四、对比分析

## 共同点

- 都能实现创建几何分布的功能；
- 都包含几何分布的一些方法，例如：均值、方差、累计分布函数（cdf）、KL散度、对数概率函数等
- 数学原理的代码实现，本质上是相同的

## 不同点

- TensorFlow 的 API 与 PyTorch 的设计思路不同，本API开发主要参考pytorch的实现逻辑，参照tensorflow。
- TensorFlow 的 API 中包含KL散度的注册，在飞桨的实现中将参考这一实现与之前拉普拉斯分布中的实现。

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.Geometric(loc, scale)
```

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `paddle.distribution.Geometric`。
基于`paddle.distribution` API基类进行开发。
class API 中的具体实现（部分方法已完成开发，故直接使用源代码）：
- `mean`计算均值；`1. / self.probs - 1.`
- `mode` 众数 ；`paddle.zeros_like(self.probs)`
- `variance`计算方差 ；`paddle.full(shape, inf, dtype=self.loc.dtype)`
- `stddev`计算标准偏差 `(1. / self.probs - 1.) / self.probs`
- `entropy` 熵计算；通过binary_cross_entropy_with_logits  API进行计算
下述方法根据相应的数学公式、pytorch源代码（代码风格与paddle有不同）、scipy源代码进行开发；需参照相关实现进行probs_to_logits, logits_to_probs,方法的开发
- `sample`随机采样；` 
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `cdf` 累积分布函数(Cumulative Distribution Function)
- `icdf` 逆累积分布函数
- 注册KL散度  参照laplace散度注册

- PS:以上代码已类似伪代码的形式给出API的实现，未经过验证。以代码实现为准。

# 六、测试和验收的考量

测试考虑的 case 如下(参考了pytorch中的Geometric单测代码)：

根据api类各个方法及特性传参的不同，把单测分成三个部分：测试分布的特性（无需额外参数）、测试分布的概率密度函数（需要传值）以及测试KL散度（需要传入一个实例）。

测试Geometric分布的特性
测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestLaplace继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

均值、方差、标准差通计算相应值的shape，对比Geometric类中相应返回值的shape，若一致即正确；


sample几何分布采样 ，熵计算通过对比scipy.stats.geom与scipy.stats.geom.entropy的值是否与类方法返回值一致验证结果的正确性。

测试用例：单测需要覆盖单一维度的Geometric分布和多维度分布情况，因此使用两种初始化参数

'one-dim': loc=parameterize.xrand((2, )), scale=parameterize.xrand((2, ));
'multi-dim': loc=parameterize.xrand((5, 5)), scale=parameterize.xrand((5, 5))。
测试Lapalce分布的概率密度函数
测试方法：该部分主要测试分布各种概率密度函数。类TestGeometricPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测），test_cdf（cdf单测），test_icdf（icdf）。以上分布在scipy.stats.Geometric中均有实现，因此给定某个输入value，对比相同参数下Geometric分布的scipy实现以及paddle实现的结果，若误差在容忍度范围内则证明实现正确。

测试用例：为不失一般性，测试使用多维位置参数和尺度参数初始化Geometric类，并覆盖int型输入及float型输入。

'value-float': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2., 5.]); * 'value-int': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2, 5]);
'value-multi-dim': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([[4., 6], [8, 2]])。
测试Lapalce分布之间的KL散度
测试方法：该部分测试两个Geometric分布之间的KL散度。类TestGeometricAndGeometricKL继承unittest.TestCase，分别实现setUp（初始化），test_kl_divergence（kl_divergence）。在scipy中scipy.stats.entropy可用来计算两个分布之间的散度。因此对比两个Geometric分布在paddle.distribution.kl_divergence下和在scipy.stats.geom下计算的散度，若结果在误差范围内，则证明该方法实现正确。

测试用例：分布1：loc=np.array([0.0]), scale=np.array([1.0]), 分布2: loc=np.array([1.0]), scale=np.array([0.5])



# 七、可行性分析及规划排期


具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.Geometric` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.Geometric` API，与飞桨2.0之后的代码风格保持一致

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.distributions.Geometric](https://pytorch.org/docs/stable/distributions.html#Geometric)





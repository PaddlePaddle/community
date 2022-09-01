# paddle.distribution.Gumbel 设计文档

| API 名称     | paddle.distribution.Gumbel        |
| ------------ | ---------------------------------- |
| 提交作者     | 李芳钰 |
| 提交时间     | 2022-09-1                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.3.0                             |
| 文件名       | 20220901_api_design_for_Gumbel.md     |

# 一、概述

## 1、相关背景

此任务的目标是在 Paddle 框架中，基于现有概率分布方案进行扩展，新增 Gumbel API， API调用 `paddle.distribution.Gumbel`。

## 2、功能目标

增加 API `paddle.distribution.Gumbel`，Gumbel 用于 Gumbel 分布的概率统计与随机采样。API具体包含如下方法：

功能：`Creates a Gumbel distribution parameterized by loc and scale.`

- `mean`计算均值；
- `variance`计算方差；
- `stddev`计算标准偏差
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `entropy` 熵计算；

## 3、意义

为 Paddle 增加用于 Gumbel 分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.Gumbel`，但是有API`paddle.distribution.Multinomial`paddle.distribution.Gumbel的开发代码风格主要参考API
- 通过反馈可以发现，代码需采用飞桨2.0之后的API，故此处不再参考Normal等API的代码风格。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.gumbel.Gumbel(loc, scale, validate_args=None)`

### 源代码

```
from numbers import Number
import math
import torch
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all, euler_constant



[docs]class Gumbel(TransformedDistribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, Number) and isinstance(scale, Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps)
        else:
            base_dist = Uniform(torch.full_like(self.loc, finfo.tiny),
                                torch.full_like(self.loc, 1 - finfo.eps))
        transforms = [ExpTransform().inv, AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
                      ExpTransform().inv, AffineTransform(loc=loc, scale=-self.scale)]
        super(Gumbel, self).__init__(base_dist, transforms, validate_args=validate_args)


[docs]    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super(Gumbel, self).expand(batch_shape, _instance=new)


    # Explicitly defining the log probability function for Gumbel due to precision issues

[docs]    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()


    @property
    def mean(self):
        return self.loc + self.scale * euler_constant

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)


[docs]    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
```

## Numpy

Numpy 中有API `numpy.random.gumbel(loc=0.0, scale=1.0, size=None)`

### 源代码
核心代码如下:
```
double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) {
  double U;

  U = 1.0 - next_double(bitgen_state);
  if (U < 1.0) {
    return loc - scale * log(-log(U));
  }
  /* Reject U == 1.0 and call again to get next value */
  return random_gumbel(bitgen_state, loc, scale);
}
```

# 四、对比分析

- 都能实现对 Gumbel 分布进行采样的功能；
- Pytorch 实现了Gumbel分布类，包含了一下属性：均值、方差、标准差等；
- Numpy 只是提供了一种对 Gumbel 分布进行采样的方法；

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.Gumbel(loc, scale)
```

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `paddle.distribution.Gumbel`。
基于`paddle.distribution` API基类进行开发。
class API 中的具体实现（部分方法已完成开发，故直接使用源代码），该 API 有两个参数：分布的位置参数self.loc 和分布的尺度参数self.scale。包含以下方法：

```
euler_constant = 0.57721566490153286060  # Euler Mascheroni Constant
```

- `mean` 计算均值: 

        return self.loc + self.scale*euler_constant
- `stddev` 计算标准差: 
        
        return (math.pi / math.sqrt(6)) * self.scale

- `variance` 计算方差: 

        return self.stddev.pow(2)

- `prob` 概率密度(包含传参value): 

        return paddle.exp(self.log_prob(value))

- `log_prob` 对数概率密度(value): 

        y = (self.loc - value) / self.scale
        return (y - paddle.exp(y)) - paddle.log(self.scale)

- `entropy` 熵计算: 

        return paddle.log(self.scale) + (1 + euler_constant)

- `sample` 随机采样(参考pytorch的实现):

         x = paddle.uniform(shape=shape, min=tiny, max=1-eps)  # tiny,eps根据dtype决定;
         transforms = [ExpTransform().inv, 
                       AffineTransform(loc=0, scale=-paddle.ones_like(self.scale)),
                       ExpTransform().inv, 
                       AffineTransform(loc=loc, scale=-self.scale)]
         for transform in transforms:
            x = transform(x)
         return x

- `rsample` 重参数化采样(直接复用sample): 

        self.sample(shape)


# 六、测试和验收的考量

根据api类各个方法及特性传参的不同，把单测分成三个部分：测试分布的特性（无需额外参数）。

1. 测试Lapalce分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestLaplace继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

  * 均值、方差、标准差通过Numpy计算相应值，对比Gumbel类中相应property的返回值，若一致即正确；
  
  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合Gumbel分布。验证策略如下：随机采样30000个Gumbel分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.laplace`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于Gumbel分布，若计算所得ks值小于0.02，则拒绝不一致假设，两者属于同一分布；
  
  * 熵计算通过对比`np.log(scale) + (1+ euler_constant)`的值是否与类方法返回值一致验证结果的正确性。

- 测试用例：单测需要覆盖单一维度的Gumbel分布和多维度分布情况，因此使用两种初始化参数:

  * 'one-dim': `loc=parameterize.xrand((2, )), scale=parameterize.xrand((2, ))`; 
  * 'multi-dim': `loc=parameterize.xrand((5, 5)), scale=parameterize.xrand((5, 5))`。

# 七、可行性分析及规划排期

具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.Gumbel` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.Gumbel` API，与飞桨2.0代码风格保持一致

# 名词解释

无

# 附件及参考资料

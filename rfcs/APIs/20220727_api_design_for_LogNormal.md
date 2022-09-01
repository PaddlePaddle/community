# paddle.distribution.LogNormal 设计文档

| API 名称     | paddle.distribution.LogNormal        |
| ------------ | ---------------------------------- |
| 提交作者     | 李文博（李长安）倪浩（TUDelftHao） |
| 提交时间     | 2022-09-01                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.3.0                             |
| 文件名       | 20220818_design_for_LogNormal.md     |

# 一、概述

## 1、相关背景

此任务的目标是在 Paddle 框架中，基于现有概率分布方案进行扩展，新增 LogNormal API， API调用 `paddle.distribution.LogNormal`。

## 2、功能目标

增加 API `paddle.distribution.LogNormal`，LogNormal 用于 LogNormal 分布的概率统计与随机采样。API具体包含如下方法：

功能：`Creates a LogNormal distribution parameterized by loc and scale.`

- `mean`计算均值；
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

为 Paddle 增加用于 LogNormal 分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.LogNormal`
- 通过反馈可以发现，代码需采用飞桨2.0之后的API，故此处不再参考Normal等API的代码风格。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.LogNormal.LogNormal(loc, scale, validate_args=None)`

### 源代码

```
class LogNormal(TransformedDistribution):
    """
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super(LogNormal, self).__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super(LogNormal, self).expand(batch_shape, _instance=new)


    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def mode(self):
        return (self.loc - self.scale.square()).exp()

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc
```



# 四、对比分析

## 共同点

- 都能实现创建对数正态分布的功能；
- 都包含对数正态分布的一些方法，例如：均值、方差、累计分布函数（cdf）、对数概率函数等
- 数学原理的代码实现，本质上是相同的


- 
# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.LogNormal(loc, scale)
```

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `paddle.distribution.LogNormal`。
基于`paddle.distribution.Normal` API基类进行开发。通俗来讲，对数正态分布就是说原未知数的分布不是正态分布，但是把数值取对数之后是正态分布。
故此API大部分方法可直接继承自`paddle.distribution.Normal`进行实现
class API 中的具体实现（部分方法已完成开发，故直接使用源代码），该api有两个参数：位置参数self.loc, 尺度参数self.scale。包含以下方法：

- `mean` 计算均值: 

        (self.loc + self.scale.pow(2) / 2).exp()

- `stddev` 计算标准差: 
        
        (2 ** 0.5) * self.scale;

    直接继承父类实现

- `variance` 计算方差: 

        (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()
    
    
- `sample` 随机采样(参考pytorch复用重参数化采样结果): 

        self.rsample(shape)

    直接继承父类实现

- `rsample` 重参数化采样: 

        self.loc - self.scale * u.sign() * paddle.log1p(-u.abs())

    其中 `u = paddle.uniform(shape=shape, min=eps - 1, max=1)`; eps根据dtype决定;

    直接继承父类实现

- `prob` 概率密度(包含传参value): 

        self.log_prob(value).exp()

    直接继承父类实现

- `log_prob` 对数概率密度(value): 

        -paddle.log(2 * self.scale) - paddle.abs(value - self.loc) / self.scale

    直接继承父类实现

- `entropy` 熵计算: 

        1 + paddle.log(2 * self.scale)

    直接继承父类实现

- `cdf` 累积分布函数(value): 

        0.5 - 0.5 * (value - self.loc).sign() * paddle.expm1(-(value - self.loc).abs() / self.scale)

    直接继承父类实现

- `icdf` 逆累积分布函数(value): 

        self.loc - self.scale * (value - 0.5).sign() * paddle.log1p(-2 * (value - 0.5).abs())

    直接继承父类实现

- `kl_divergence` 两个LogNormal分布之间的kl散度(other--LogNormal类的一个实例):

        (self.scale * paddle.exp(paddle.abs(self.loc - other.loc) / self.scale) + paddle.abs(self.loc - other.loc)) / other.scale + paddle.log(other.scale / self.scale) - 1
     参考文献：https://openaccess.thecvf.com/content/CVPR2021/supplemental/Meyer_An_Alternative_Probabilistic_CVPR_2021_supplemental.pdf 

    同时在`paddle/distribution/kl.py` 中注册`_kl_LogNormal_LogNormal`函数，使用时可直接调用kl_divergence计算LogNormal分布之间的kl散度。
  

# 六、测试和验收的考量

根据api类各个方法及特性传参的不同，把单测分成三个部分：测试分布的特性（无需额外参数）、测试分布的概率密度函数（需要传值）以及测试KL散度（需要传入一个实例）。

1. 测试对数正态分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestLogNormal继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

  * 均值、方差、标准差通过Numpy计算相应值，对比LogNormal类中相应property的返回值，若一致即正确；
  
  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合LogNormal分布。验证策略如下：随机采样30000个LogNormal分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.LogNormal`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于LogNormal分布，若计算所得ks值小于0.02，则拒绝不一致假设，两者属于同一分布；
  
  * 熵计算通过对比`scipy.stats.LogNormal.entropy`的值是否与类方法返回值一致验证结果的正确性。

- 测试用例：单测需要覆盖单一维度的LogNormal分布和多维度分布情况，因此使用两种初始化参数

  * 'one-dim': `loc=parameterize.xrand((2, )), scale=parameterize.xrand((2, ))`; 
  * 'multi-dim': loc=parameterize.xrand((5, 5)), scale=parameterize.xrand((5, 5))。


2. 测试对数正态分布的概率密度函数

- 测试方法：该部分主要测试分布各种概率密度函数。类TestLogNormalPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测），test_cdf（cdf单测），test_icdf（icdf）。以上分布在`scipy.stats.LogNormal`中均有实现，因此给定某个输入value，对比相同参数下LogNormal分布的scipy实现以及paddle实现的结果，若误差在容忍度范围内则证明实现正确。

- 测试用例：为不失一般性，测试使用多维位置参数和尺度参数初始化LogNormal类，并覆盖int型输入及float型输入。
  * 'value-float': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2., 5.])`; * 'value-int': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2, 5])`; 
  * 'value-multi-dim': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([[4., 6], [8, 2]])`。

3. 测试Lapalce分布之间的KL散度

- 测试方法：该部分测试两个LogNormal分布之间的KL散度。类TestLogNormalAndLogNormalKL继承unittest.TestCase，分别实现setUp（初始化），test_kl_divergence（kl_divergence）。在scipy中`scipy.stats.entropy`可用来计算两个分布之间的散度。因此对比两个LogNormal分布在`paddle.distribution.kl_divergence`下和在scipy.stats.LogNormal下计算的散度，若结果在误差范围内，则证明该方法实现正确。

- 测试用例：分布1：`loc=np.array([0.0]), scale=np.array([1.0])`, 分布2: `loc=np.array([1.0]), scale=np.array([0.5])`



# 七、可行性分析及规划排期

具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.LogNormal` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.LogNormal` API，与飞桨2.0代码风格保持一致

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/stable/distributions.html?highlight=lognormal#torch.distributions.log_normal.LogNormal)


## Paddle

[paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Normal_cn.html#normal)
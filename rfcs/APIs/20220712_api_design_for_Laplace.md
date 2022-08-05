# paddle.distribution.Laplace 设计文档

| API 名称     | paddle.distribution.Laplace        |
| ------------ | ---------------------------------- |
| 提交作者     | 李文博（李长安）倪浩（TUDelftHao） |
| 提交时间     | 2022-07-12                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.3.0                             |
| 文件名       | 20220706_design_for_Laplace.md     |

# 一、概述

## 1、相关背景

此任务的目标是在 Paddle 框架中，基于现有概率分布方案进行扩展，新增 Laplace API， API调用 `paddle.distribution.Laplace`。

## 2、功能目标

增加 API `paddle.distribution.Laplace`，Laplace 用于 Laplace 分布的概率统计与随机采样。API具体包含如下方法：

功能：`Creates a Laplace distribution parameterized by loc and scale.`

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

为 Paddle 增加用于 Laplace 分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.Laplace`，但是有API`paddle.distribution.Multinomial`paddle.distribution.Laplace的开发代码风格主要参考API
- 通过反馈可以发现，代码需采用飞桨2.0之后的API，故此处不再参考Normal等API的代码风格。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.laplace.Laplace(loc, scale, validate_args=None)`

### 源代码

```
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


[docs]class Laplace(Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return 2 * self.scale.pow(2)

    @property
    def stddev(self):
        return (2 ** 0.5) * self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Laplace, self).__init__(batch_shape, validate_args=validate_args)

[docs]    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


[docs]    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(-u.abs().clamp(min=finfo.tiny))
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(-(value - self.loc).abs() / self.scale)


    def icdf(self, value):
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())


    def entropy(self):
        return 1 + torch.log(2 * self.scale)
```

## TensorFlow

TensorFlow 中包含 class API `tf.compat.v1.distributions.Laplace`

### 源代码

```
@tf_export(v1=["distributions.Laplace"])
class Laplace(distribution.Distribution):
  """The Laplace distribution with location `loc` and `scale` parameters.
  #### Mathematical details
  The probability density function (pdf) of this distribution is,
  ```none
  pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
  Z = 2 sigma

  where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.
  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."
  The Lpalce distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,
  ```none
  X ~ Laplace(loc=0, scale=1)
  Y = loc + scale * X

  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="Laplace"):
    """Construct Laplace distribution with parameters `loc` and `scale`.
    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).
    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      TypeError: if `loc` and `scale` are of different dtype.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[loc, scale]) as name:
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype([self._loc, self._scale])
      super(Laplace, self).__init__(
          dtype=self._loc.dtype,
          reparameterization_type=distribution.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[self._loc, self._scale],
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc), array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(), self.scale.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.TensorShape([])

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    # Uniform variates must be sampled from the open-interval `(-1, 1)` rather
    # than `[-1, 1)`. In the case of `(0, 1)` we'd use
    # `np.finfo(self.dtype.as_numpy_dtype).tiny` because it is the smallest,
    # positive, "normal" number. However, the concept of subnormality exists
    # only at zero; here we need the smallest usable number larger than -1,
    # i.e., `-1 + eps/2`.
    uniform_samples = random_ops.random_uniform(
        shape=shape,
        minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                            self.dtype.as_numpy_dtype(0.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    return (self.loc - self.scale * math_ops.sign(uniform_samples) *
            math_ops.log1p(-math_ops.abs(uniform_samples)))

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return special_math.log_cdf_laplace(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_cdf_laplace(-self._z(x))

  def _cdf(self, x):
    z = self._z(x)
    return (0.5 + 0.5 * math_ops.sign(z) *
            (1. - math_ops.exp(-math_ops.abs(z))))

  def _log_unnormalized_prob(self, x):
    return -math_ops.abs(self._z(x))

  def _log_normalization(self):
    return math.log(2.) + math_ops.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
    scale = self.scale + array_ops.zeros_like(self.loc)
    return math.log(2.) + 1. + math_ops.log(scale)

  def _mean(self):
    return self.loc + array_ops.zeros_like(self.scale)

  def _stddev(self):
    return math.sqrt(2.) * self.scale + array_ops.zeros_like(self.loc)

  def _median(self):
    return self._mean()

  def _mode(self):
    return self._mean()

  def _z(self, x):
    return (x - self.loc) / self.scale
```

# 四、对比分析

## 共同点

- 都能实现创建拉普拉斯分布的功能；
- 都包含拉普拉斯分布的一些方法，例如：均值、方差、累计分布函数（cdf）、对数概率函数等
- 数学原理的代码实现，本质上是相同的

## 不同点

- TensorFlow 的 API 与 PyTorch 的设计思路不同，本API开发主要参考pytorch的实现逻辑，参照tensorflow。

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.Laplace(loc, scale)
```

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `paddle.distribution.Laplace`。
基于`paddle.distribution` API基类进行开发。
class API 中的具体实现（部分方法已完成开发，故直接使用源代码），该api有两个参数：位置参数self.loc, 尺度参数self.scale。包含以下方法：

- `mean` 计算均值: 

        self.loc
- `stddev` 计算标准差: 
        
        (2 ** 0.5) * self.scale;

- `variance` 计算方差: 

        self.stddev.pow(2)

- `sample` 随机采样(参考pytorch复用重参数化采样结果): 

        self.rsample(shape)

- `rsample` 重参数化采样: 

        self.loc - self.scale * u.sign() * paddle.log1p(-u.abs())
    其中 `u = paddle.uniform(shape=shape, min=eps - 1, max=1)`; eps根据dtype决定;
- `prob` 概率密度(包含传参value): 

        self.log_prob(value).exp()

    直接继承父类实现

- `log_prob` 对数概率密度(value): 

        -paddle.log(2 * self.scale) - paddle.abs(value - self.loc) / self.scale

- `entropy` 熵计算: 

        1 + paddle.log(2 * self.scale)

- `cdf` 累积分布函数(value): 

        0.5 - 0.5 * (value - self.loc).sign() * paddle.expm1(-(value - self.loc).abs() / self.scale)

- `icdf` 逆累积分布函数(value): 

        self.loc - self.scale * (value - 0.5).sign() * paddle.log1p(-2 * (value - 0.5).abs())

- `kl_divergence` 两个Laplace分布之间的kl散度(other--Laplace类的一个实例):

        (self.scale * paddle.exp(paddle.abs(self.loc - other.loc) / self.scale) + paddle.abs(self.loc - other.loc)) / other.scale + paddle.log(other.scale / self.scale) - 1
     参考文献：https://openaccess.thecvf.com/content/CVPR2021/supplemental/Meyer_An_Alternative_Probabilistic_CVPR_2021_supplemental.pdf 

    同时在`paddle/distribution/kl.py` 中注册`_kl_laplace_laplace`函数，使用时可直接调用kl_divergence计算laplace分布之间的kl散度。
  

# 六、测试和验收的考量

根据api类各个方法及特性传参的不同，把单测分成三个部分：测试分布的特性（无需额外参数）、测试分布的概率密度函数（需要传值）以及测试KL散度（需要传入一个实例）。

1. 测试Lapalce分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestLaplace继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

  * 均值、方差、标准差通过Numpy计算相应值，对比Laplace类中相应property的返回值，若一致即正确；
  
  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合laplace分布。验证策略如下：随机采样30000个laplace分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.laplace`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于laplace分布，若计算所得ks值小于0.02，则拒绝不一致假设，两者属于同一分布；
  
  * 熵计算通过对比`scipy.stats.laplace.entropy`的值是否与类方法返回值一致验证结果的正确性。

- 测试用例：单测需要覆盖单一维度的Laplace分布和多维度分布情况，因此使用两种初始化参数

  * 'one-dim': `loc=parameterize.xrand((2, )), scale=parameterize.xrand((2, ))`; 
  * 'multi-dim': loc=parameterize.xrand((5, 5)), scale=parameterize.xrand((5, 5))。


2. 测试Lapalce分布的概率密度函数

- 测试方法：该部分主要测试分布各种概率密度函数。类TestLaplacePDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测），test_cdf（cdf单测），test_icdf（icdf）。以上分布在`scipy.stats.laplace`中均有实现，因此给定某个输入value，对比相同参数下Laplace分布的scipy实现以及paddle实现的结果，若误差在容忍度范围内则证明实现正确。

- 测试用例：为不失一般性，测试使用多维位置参数和尺度参数初始化Laplace类，并覆盖int型输入及float型输入。
  * 'value-float': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2., 5.])`; * 'value-int': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2, 5])`; 
  * 'value-multi-dim': `loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([[4., 6], [8, 2]])`。

3. 测试Lapalce分布之间的KL散度

- 测试方法：该部分测试两个Laplace分布之间的KL散度。类TestLaplaceAndLaplaceKL继承unittest.TestCase，分别实现setUp（初始化），test_kl_divergence（kl_divergence）。在scipy中`scipy.stats.entropy`可用来计算两个分布之间的散度。因此对比两个Laplace分布在`paddle.distribution.kl_divergence`下和在scipy.stats.laplace下计算的散度，若结果在误差范围内，则证明该方法实现正确。

- 测试用例：分布1：`loc=np.array([0.0]), scale=np.array([1.0])`, 分布2: `loc=np.array([1.0]), scale=np.array([0.5])`



# 七、可行性分析及规划排期

具体规划为

- 阶段一：完成API功能开发（目前已完成，代码风格需从1.8升级至2.0以上）
- 阶段二：完成 `paddle.distribution.Laplace` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.Laplace` API，与飞桨2.0代码风格保持一致

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.distributions.laplace.Laplace](https://pytorch.org/docs/stable/distributions.html#laplace)

## TensorFlow

[tf.compat.v1.distributions.Laplace](https://www.tensorflow.org/api_docs/python/tf/compat/v1/distributions/Laplace)

## Paddle

[paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Normal_cn.html#normal)

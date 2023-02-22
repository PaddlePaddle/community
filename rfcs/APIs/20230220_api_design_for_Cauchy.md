# paddle.distribution.Cauchy 设计文档

| API 名称     | paddle.distribution.Cauchy        |
| ------------ | ---------------------------------- |
| 提交作者     | 李文博（李长安）倪浩（TUDelftHao） |
| 提交时间     | 2023-02-20                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.4.2                             |
| 文件名       | 20230220_design_for_Cauchy_distribution.md     |

# 一、概述

## 1、相关背景

此任务的目标是在 Paddle 框架中，基于现有概率分布方案进行扩展，新增 Laplace API， API调用 `paddle.distribution.Cauchy`。

## 2、功能目标

增加 API `paddle.distribution.Cauchy`，Cauchy 用于 Cauchy 分布的概率统计与随机采样。API具体包含如下方法：

功能：`Creates a Laplace distribution parameterized by loc and scale.`

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

为 Paddle 增加用于 Laplace 分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状 

- 目前 飞桨没有 API `paddle.distribution.Cauchy`，

- 调研 Paddle 及业界实现惯例，并且代码风格及设计思路与已有概率分布保持一致代码，需采用飞桨2.0之后的API，故依赖飞桨版本V2.4.2 。

- PS：已经参与Laplace的API贡献，故此部分较为熟悉。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.cauchy.Cauchy(loc, scale, validate_args=None)`


### 源代码

```
import math
from torch._six import inf, nan
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

__all__ = ['Cauchy']

class Cauchy(Distribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Cauchy, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    @property
    def mean(self):
        return torch.full(self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device)

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.full(self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -math.log(math.pi) - self.scale.log() - (1 + ((value - self.loc) / self.scale)**2).log()


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5


    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc


    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
```
## TensorFlow


```
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Cauchy',
]


class Cauchy(distribution.AutoCompositeTensorDistribution):
  
  #### Examples
  Examples of initialization of one or a batch of distributions.
  ```python
  tfd = tfp.distributions
  # Define a single scalar Cauchy distribution.
  dist = tfd.Cauchy(loc=0., scale=3.)
  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)
  # Define a batch of two scalar valued Cauchy distributions.
  dist = tfd.Cauchy(loc=[1, 2.], scale=[11, 22.])
  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])
  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  # Arguments are broadcast when possible.
  # Define a batch of two scalar valued Cauchy distributions.
  # Both have median 1, but different scales.
  dist = tfd.Cauchy(loc=1., scale=[11, 22.])
  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.)


  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Cauchy'):
    """Construct Cauchy distributions.
    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).
    Args:
      loc: Floating point tensor; the modes of the distribution(s).
      scale: Floating point tensor; the half-widths of the distribution(s) at
        their half-maximums. Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype([self._loc, self._scale])
      super(Cauchy, self).__init__(
          dtype=self._scale.dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(loc=loc, scale=scale)
    shape = ps.concat([[n], batch_shape], 0)
    probs = samplers.uniform(
        shape=shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    return self._quantile(probs, loc=loc, scale=scale)

  def _log_prob(self, x):
    npdt = dtype_util.as_numpy_dtype(self.dtype)
    scale = tf.convert_to_tensor(self.scale)
    log_unnormalized_prob = -tf.math.log1p(tf.square(self._z(x, scale=scale)))
    log_normalization = npdt(np.log(np.pi)) + tf.math.log(scale)
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    return tf.atan(self._z(x)) / np.pi + 0.5

  def _log_cdf(self, x):
    return tf.math.log1p(2 / np.pi * tf.atan(self._z(x))) - np.log(2)

  def _entropy(self):
    h = np.log(4 * np.pi) + tf.math.log(self.scale)
    return h * tf.ones_like(self.loc)

  def _quantile(self, p, loc=None, scale=None):
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    scale = tf.convert_to_tensor(self.scale if scale is None else scale)
    return loc + scale * tf.tan(np.pi * (p - 0.5))

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _z(self, x, loc=None, scale=None):
    """Standardize input `x`."""
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    scale = tf.convert_to_tensor(self.scale if scale is None else scale)
    with tf.name_scope('standardize'):
      return (x - loc) / scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with tf.name_scope('reconstruct'):
      return z * self.scale + self.loc

  def _mean(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      raise ValueError('`mean` is undefined for Cauchy distribution.')

  def _stddev(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      raise ValueError('`stddev` is undefined for Cauchy distribution.')

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector (consider one that
    # transforms away the heavy tails).
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Cauchy, Cauchy)
def _kl_cauchy_cauchy(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Cauchy.
  Note that this KL divergence is symmetric in its arguments.
  Args:
    a: instance of a Cauchy distribution object.
    b: instance of a Cauchy distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_cauchy_cauchy'`).
  Returns:
    kl_div: Batchwise KL(a || b)
  #### References
  [1] Frederic Chyzak and Frank Nielsen. A closed-form formula for the
  Kullback-Leibler divergence between Cauchy distributions.
  https://arxiv.org/abs/1905.10965
  
  with tf.name_scope(name or 'kl_cauchy_cauchy'):
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    b_loc = tf.convert_to_tensor(b.loc)
    scale_sum_square = tf.math.square(a_scale + b_scale)
    loc_diff_square = tf.math.squared_difference(a.loc, b_loc)

    return (tf.math.log(scale_sum_square + loc_diff_square) -
            np.log(4.) - tf.math.log(a_scale) - tf.math.log(b_scale))

```


# 四、对比分析

## 共同点

- 都能实现创建柯西分布的功能；
- 都包含柯西分布的一些方法，例如：均值、方差、累计分布函数（cdf）、KL散度、对数概率函数等
- 数学原理的代码实现，本质上是相同的

## 不同点

- TensorFlow 的 API 与 PyTorch 的设计思路不同，本API开发主要参考pytorch的实现逻辑，参照tensorflow。
- TensorFlow 的 API 中包含KL散度的注册，在飞桨的实现中将参考这一实现与之前拉普拉斯分布中的实现。

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.Cauchy(loc, scale)
```

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `paddle.distribution.Cauchy`。
基于`paddle.distribution` API基类进行开发。
class API 中的具体实现（部分方法已完成开发，故直接使用源代码）：
- `mean`计算均值；`paddle.full(shape, nan, dtype=self.loc.dtype)`
- `mode` 众数 ；`self.loc`
- `variance`计算方差 ；`paddle.full(shape, inf, dtype=self.loc.dtype)`
- `stddev`计算标准偏差 `(2 ** 0.5) * self.scale`
- `sample`随机采样；` with paddle.no_grad():
            return self.rsample(shape)`
- `rsample` 重参数化采样；
    `def rsample(self, sample_shape=paddle.Size()):shape = sample_shape eps = self.loc.new(shape).cauchy_() return self.loc + eps * self.scale`
        
        
- `prob` 概率密度；`-math.log(math.pi) - self.scale.log() - (1 + ((value - self.loc) / self.scale)**2).log()`
- `log_prob`对数概率密度；`subtract(-nn.log(2 * self.scale), paddle.abs(value - self.loc) / self.scale)`
- `entropy` 熵计算；`math.log(4 * math.pi) + self.scale.log()`
- `cdf` 累积分布函数(Cumulative Distribution Function)`paddle.atan((value - self.loc) / self.scale) / math.pi + 0.5)`
- `icdf` 逆累积分布函数`paddle.tan(math.pi * (value - 0.5)) * self.scale + self.loc`
- 注册KL散度  参照laplace散度注册

- PS:以上代码已类似伪代码的形式给出API的实现，未经过验证。以代码实现为准。

# 六、测试和验收的考量

测试考虑的 case 如下(参考pytorch中的单测代码)：

根据api类各个方法及特性传参的不同，把单测分成三个部分：测试分布的特性（无需额外参数）、测试分布的概率密度函数（需要传值）以及测试KL散度（需要传入一个实例）。

测试Cauchy分布的特性

测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestLaplace继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

考虑到柯西分布的特殊性：数学期望与方差均不存在
均值、方差等通过Numpy计算相应值（inf，nan），对比Laplace类中相应property的返回值，若一致即正确；

采样方法 验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合Cauchy分布。验证策略如下：随机采样30000个Cauchy分布下的样本值，并比较同分布下numpy返回值，检查是否在合理误差范围内
熵计算通过对比scipy.stats.Cauchy.entropy的值是否与类方法返回值一致验证结果的正确性。

测试用例：单测需要覆盖单一维度的Cauchy分布和多维度分布情况，因此使用两种初始化参数

'one-dim': loc=parameterize.xrand((2, )), scale=parameterize.xrand((2, ));
'multi-dim': loc=parameterize.xrand((5, 5)), scale=parameterize.xrand((5, 5))。
测试Lapalce分布的概率密度函数
测试方法：该部分主要测试分布各种概率密度函数。类TestCauchyPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测），test_cdf（cdf单测），test_icdf（icdf）。以上分布在scipy.stats.Cauchy中均有实现，因此给定某个输入value，对比相同参数下Cauchy分布的scipy实现以及paddle实现的结果，若误差在容忍度范围内则证明实现正确。

测试用例：为不失一般性，测试使用多维位置参数和尺度参数初始化Cauchy类，并覆盖int型输入及float型输入。

'value-float': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2., 5.]); * 'value-int': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([2, 5]);
'value-multi-dim': loc=np.array([0.2, 0.3]), scale=np.array([2, 3]), value=np.array([[4., 6], [8, 2]])。
测试Lapalce分布之间的KL散度
测试方法：该部分测试两个Cauchy分布之间的KL散度。类TestCauchyAndCauchyKL继承unittest.TestCase，分别实现setUp（初始化），test_kl_divergence（kl_divergence）。在scipy中scipy.stats.entropy可用来计算两个分布之间的散度。因此对比两个Cauchy分布在paddle.distribution.kl_divergence下和在scipy.stats.Cauchy下计算的散度，若结果在误差范围内，则证明该方法实现正确。

测试用例：分布1：loc=np.array([0.0]), scale=np.array([1.0]), 分布2: loc=np.array([1.0]), scale=np.array([0.5])



# 七、可行性分析及规划排期


具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.Cauchy` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.Cauchy` API，与飞桨2.0之后的代码风格保持一致

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.distributions.Cauchy](https://pytorch.org/docs/stable/distributions.html#Cauchy)

[Pytorch柯西分布单测文件](https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py#L1602)

## TensorFlow

[tfp.distributions.Cauchy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Cauchy)



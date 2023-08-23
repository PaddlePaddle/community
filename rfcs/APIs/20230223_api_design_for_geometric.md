| API 名称 | paddle.distribution.Geometric        |
| --- |--------------------------------------|
| 提交作者 | 王勇森(dasenCoding)                     |
| 提交时间 | 2023-02-23                           |
| 版本号 | V1.0.0                               |
| 依赖飞桨版本 | V2.4.1                               |
| 文件名 | 20230220_api_design_for_geometric.md |

# 一、概述
## 1、相关背景
几何分布（`Geometric distribution`）是离散型概率分布。其中一种定义为：在n次伯努利试验中，试验k次才得到第一次成功的机率。详细地说，是：前k-1次皆失败，第k次成功的概率。几何分布是帕斯卡分布当r=1时的特例。[1]

随着n增大呈等比级数变化，等比级数又称几何级数。这可能和以前几何学中无限分割图形得到的级数有关。

`Geometric distribution` 应用条件：进行一系列独立试验，每一次试验活成功或失败，每一次试验的成功概率相同，即：为了取得第一次成功，需要进行多少次试验。

目前 Paddle 框架中没有实现 Geometric 分布。所以此任务的目标是在 Paddle 框架中，基于现有概率分布方案，在其基础上进行扩展，新增 Geometric API，API 的调用路径为： `paddle.distribution.Geometric`。
> 参考资料
> 
> [1]：百度百科 https://baike.baidu.com/item/%E5%87%A0%E4%BD%95%E5%88%86%E5%B8%83/10676983?fr=aladdin


## 2、功能目标

为 paddle 框架增加 API  `paddle.distribution.Geometric`，Geometric 表示几何分布，用于几何分布的概率统计与随机采样。API中包括了如下方法：

- `mean`计算均值；
- `mode`计算众数
- `variance`计算方差 ；
- `sample`随机采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `cdf`累计分布函数；
- `entropy`熵计算；
- `kl` 两个分布间的kl散度；

上述方法可能无法全部支持，需要设计中说明不支持原因，抛出NotImplementedError异常即可。
> 注：Geometric 无法进行重参数化采样 rsample，目前的理解为(可能存在认知漏洞)：Geometric 为离散型随机分布，重参数化采样无法后仍无法进行梯度计算。

## 3、意义

为 Paddle 增加几何分布的概率统计与随机采样函数，丰富 `paddle.distribution` 下的 API，丰富 paddle 框架。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.Geometric`
- API `paddle.distribution.Beta`的代码开发风格可以作为`paddle.distribution.Geometric` 的主要参考。


# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.geometric.Geometric(probs=None, logits=None, validate_args=None)`

主要参数变量：


- **probs  (**_Number, Tensor_**) – the probability of sampling 1. Must be in range (0, 1]**

- **logits  (**_Number, Tensor_**) – the log-odds of sampling 1**

### 源代码

```python
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property
from torch.nn.functional import binary_cross_entropy_with_logits

__all__ = ['Geometric']

class Geometric(Distribution):
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

## TensorFlow

TensorFlow 中包含 API `tfp.distributions.Geometric`。

主要参数变量包括：
```python
tfp.distributions.Geometric(
    logits=None,
    probs=None,
    force_probs_to_zero_outside_support=False,
    validate_args=False,
    allow_nan_stats=True,
    name='Geometric'
)
```

### 源代码

```python
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic

class Geometric(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
    def __init__(self,
                 logits=None,
                 probs=None,
                 force_probs_to_zero_outside_support=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Geometric'):
      """Construct Geometric distributions.
      Args:
        logits: Floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >= 0`
          indicates the number of batch dimensions. Each entry represents logits
          for the probability of success for independent Geometric distributions
          and must be in the range `(-inf, inf]`. Only one of `logits` or `probs`
          should be specified.
        probs: Positive floating-point `Tensor` with shape `[B1, ..., Bb]`
          where `b >= 0` indicates the number of batch dimensions. Each entry
          represents the probability of success for independent Geometric
          distributions and must be in the range `(0, 1]`. Only one of `logits`
          or `probs` should be specified.
        force_probs_to_zero_outside_support: Python `bool`. When `True`, negative
          and non-integer values are evaluated "strictly": `log_prob` returns
          `-inf`, `prob` returns `0`, and `cdf` and `sf` correspond.  When
          `False`, the implementation is free to save computation (and TF graph
          size) by evaluating something that matches the Geometric pmf at integer
          values `k` but produces an unrestricted result on other inputs. In the
          case of Geometric distribution, the `log_prob` formula in this case
          happens to be the continuous function `k * log(1 - probs) + log(probs)`.
          Note that this function is not a normalized probability log-density.
          Default value: `False`.
        validate_args: Python `bool`, default `False`. When `True` distribution
          parameters are checked for validity despite possibly degrading runtime
          performance. When `False` invalid inputs may silently render incorrect
          outputs.
        allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
          (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
          result is undefined. When `False`, an exception is raised if one or
          more of the statistic's batch members are undefined.
        name: Python `str` name prefixed to Ops created by this class.
      """
    
      parameters = dict(locals())
      if (probs is None) == (logits is None):
        raise ValueError('Must pass probs or logits, but not both.')
      with tf.name_scope(name) as name:
        dtype = dtype_util.common_dtype([logits, probs], dtype_hint=tf.float32)
        self._probs = tensor_util.convert_nonref_to_tensor(
            probs, dtype=dtype, name='probs')
        self._logits = tensor_util.convert_nonref_to_tensor(
            logits, dtype=dtype, name='logits')
        self._force_probs_to_zero_outside_support = (
            force_probs_to_zero_outside_support)
        super(Geometric, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)
    
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
      return dict(
          logits=parameter_properties.ParameterProperties(),
          probs=parameter_properties.ParameterProperties(
              default_constraining_bijector_fn=softmax_centered_bijector
              .SoftmaxCentered,
              is_preferred=False))
    
    @property
    def logits(self):
      """Input argument `logits`."""
      return self._logits
    
    @property
    def probs(self):
      """Input argument `probs`."""
      return self._probs
    
    @property
    def force_probs_to_zero_outside_support(self):
      """Return 0 probabilities on non-integer inputs."""
      return self._force_probs_to_zero_outside_support
    
    def _event_shape_tensor(self):
      return tf.constant([], dtype=tf.int32)
    
    def _event_shape(self):
      return tf.TensorShape([])
    
    def _sample_n(self, n, seed=None):
      # Uniform variates must be sampled from the open-interval `(0, 1)` rather
      # than `[0, 1)`. To do so, we use
      # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny`
      # because it is the smallest, positive, 'normal' number. A 'normal' number
      # is such that the mantissa has an implicit leading 1. Normal, positive
      # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
      # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
      # 0.
      probs = self._probs_parameter_no_checks()
      sampled = samplers.uniform(
          ps.concat([[n], ps.shape(probs)], 0),
          minval=np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny,
          maxval=1.,
          seed=seed,
          dtype=self.dtype)
    
      return tf.floor(tf.math.log(sampled) / tf.math.log1p(-probs))
    
    def _log_survival_function(self, x):
      probs = self._probs_parameter_no_checks()
      if not self.validate_args:
        # Whether or not x is integer-form, the following is well-defined.
        # However, scipy takes the floor, so we do too.
        x = tf.floor(x)
      return tf.where(
          x < 0.,
          dtype_util.as_numpy_dtype(x.dtype)(0.),
          (1. + x) * tf.math.log1p(-probs))
    
    def _log_cdf(self, x):
      probs = self._probs_parameter_no_checks()
      if not self.validate_args:
        # Whether or not x is integer-form, the following is well-defined.
        # However, scipy takes the floor, so we do too.
        x = tf.floor(x)
      return tf.where(
          x < 0.,
          dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
          generic.log1mexp((1. + x) * tf.math.log1p(-probs)))
    
    def _log_prob(self, x):
      probs = self._probs_parameter_no_checks()
      if not self.validate_args:
        # For consistency with cdf, we take the floor.
        x = tf.floor(x)
    
      log_probs = tf.math.xlog1py(x, -probs) + tf.math.log(probs)
    
      if self.force_probs_to_zero_outside_support:
        # Set log_prob = -inf when value is less than 0, ie prob = 0.
        log_probs = tf.where(
            x < 0.,
            dtype_util.as_numpy_dtype(x.dtype)(-np.inf),
            log_probs)
      return log_probs
    
    def _entropy(self):
      logits, probs = self._logits_and_probs_no_checks()
      if not self.validate_args:
        assertions = []
      else:
        assertions = [assert_util.assert_less(
            probs, dtype_util.as_numpy_dtype(self.dtype)(1.),
            message='Entropy is undefined when logits = inf or probs = 1.')]
      with tf.control_dependencies(assertions):
        # Claim: entropy(p) = softplus(s)/p - s
        # where s=logits and p=probs.
        #
        # Proof:
        #
        # entropy(p)
        # := -[(1-p)log(1-p) + plog(p)]/p
        # = -[log(1-p) + plog(p/(1-p))]/p
        # = -[-softplus(s) + ps]/p
        # = softplus(s)/p - s
        #
        # since,
        # log[1-sigmoid(s)]
        # = log[1/(1+exp(s)]
        # = -log[1+exp(s)]
        # = -softplus(s)
        #
        # using the fact that,
        # 1-sigmoid(s) = sigmoid(-s) = 1/(1+exp(s))
        return tf.math.softplus(logits) / probs - logits
    
    def _mean(self):
      return tf.exp(-self._logits_parameter_no_checks())
    
    def _variance(self):
      logits, probs = self._logits_and_probs_no_checks()
      return tf.exp(-logits) / probs
    
    def _mode(self):
      return tf.zeros(self.batch_shape_tensor(), dtype=self.dtype)
    
    def logits_parameter(self, name=None):
      """Logits computed from non-`None` input arg (`probs` or `logits`)."""
      with self._name_and_control_scope(name or 'logits_parameter'):
        if self._logits is None:
          return tf.math.log(self._probs) - tf.math.log1p(-self._probs)
        return tf.identity(self._logits)
    
    def probs_parameter(self, name=None):
      """Probs computed from non-`None` input arg (`probs` or `logits`)."""
      with self._name_and_control_scope(name or 'probs_parameter'):
        if self._logits is None:
          return tf.identity(self._probs)
        return tf.math.sigmoid(self._logits)
    
    def _logits_parameter_no_checks(self):
      if self._logits is None:
        probs = tf.convert_to_tensor(self._probs)
        return tf.math.log(probs) - tf.math.log1p(-probs)
      return tensor_util.identity_as_tensor(self._logits)
    
    def _probs_parameter_no_checks(self):
      if self._logits is None:
        return tensor_util.identity_as_tensor(self._probs)
      return tf.math.sigmoid(self._logits)
    
    def _logits_and_probs_no_checks(self):
      if self._logits is None:
        probs = tf.convert_to_tensor(self._probs)
        logits = tf.math.log(probs) - tf.math.log1p(-probs)
      else:
        logits = tf.convert_to_tensor(self._logits)
        probs = tf.math.sigmoid(logits)
      return logits, probs
    
    def _default_event_space_bijector(self):
      return
    
    def _sample_control_dependencies(self, x):
      assertions = []
      if not self.validate_args:
        return assertions
      assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
      return assertions
    
    def _parameter_control_dependencies(self, is_init):
      if not self.validate_args:
        return []
      assertions = []
      if self._probs is not None:
        if is_init != tensor_util.is_ref(self._probs):
          probs = tf.convert_to_tensor(self._probs)
          assertions.append(assert_util.assert_positive(
              probs, message='Argument `probs` must be positive.'))
          assertions.append(assert_util.assert_less_equal(
              probs, dtype_util.as_numpy_dtype(self.dtype)(1.),
              message='Argument `probs` must be less than or equal to 1.'))
      return assertions
    
    @classmethod
    def _maximum_likelihood_parameters(cls, value):
      return {'logits': -tf.math.log(tf.reduce_mean(value, axis=0))}
```

## Numpy

Numpy 中包含 API `numpy.random.Generator.geometric`。

主要参数包括：

- **p：float or array_like of floats**

The probability of success of an individual trial.

- **sizeint or tuple of ints, optional**

Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None (default), a single value is returned if p is a scalar. Otherwise, np.array(p).size samples are drawn.


### 源代码

```python
def geometric(self, p, size=None): # real signature unknown; restored from __doc__
    """
    geometric(p, size=None)
    
            Draw samples from the geometric distribution.
    
            Bernoulli trials are experiments with one of two outcomes:
            success or failure (an example of such an experiment is flipping
            a coin).  The geometric distribution models the number of trials
            that must be run in order to achieve success.  It is therefore
            supported on the positive integers, ``k = 1, 2, ...``.
    
            The probability mass function of the geometric distribution is
    
            .. math:: f(k) = (1 - p)^{k - 1} p
    
            where `p` is the probability of success of an individual trial.
    
            Parameters
            ----------
            p : float or array_like of floats
                The probability of success of an individual trial.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.  If size is ``None`` (default),
                a single value is returned if ``p`` is a scalar.  Otherwise,
                ``np.array(p).size`` samples are drawn.
    
            Returns
            -------
            out : ndarray or scalar
                Drawn samples from the parameterized geometric distribution.
    
            Examples
            --------
            Draw ten thousand values from the geometric distribution,
            with the probability of an individual success equal to 0.35:
    
            >>> z = np.random.default_rng().geometric(p=0.35, size=10000)
    
            How many trials succeeded after a single run?
    
            >>> (z == 1).sum() / 10000.
            0.34889999999999999 # random
    """
    pass
```


# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

## 共同点

1. 三者均实现了 Geometric 分布功能，同时数学原理保持一致，仅是各自实现的方式不同。
2. PyTorch 和 TensorFlow 进行 Geometric 初始化时，均使用伯努利实验成功的概率 probs 以及采样的对数概率 logits 进行初始化同时给定了验证参数。
3. PyTorch 和 TensorFlow 进行 Geometric 初始化时，probs 和 logits 只能二选一，并将另外一个置为 None。
4. PyTorch 和 TensorFlow 均实现了包括均值（mean）、方差（variance）、概率密度函数（probs）、熵（entropy）、对数概率密度（log_prob），模式（mode）以及取样（sample）等方法和属性。
6. 三者均实现了对输入参数的检查，虽然各自所实现的方式不同，但总体的功能是类似的。

## 不同点

1. tensorflow 作为更早发展的深度学习框架，相比于其他两者，提供了更多的接口方法，包括但不限于下面：
```python
covariance(
    name='covariance', **kwargs
)
# 协方差仅为非标量事件而定义。
```
```python
kl_divergence(
    other, name='kl_divergence'
)
# 计算两个几何分布之间kl值。
```
```python
cdf(
    value, name='cdf', **kwargs
)
# 累计分布函数。
```

```python
log_cdf(
    value, name='log_cdf', **kwargs
)
# 对数累计分布函数。
```

2. numpy 不同于 pytorch 与 tensorflow，其仅提供了初始化参数需要的条件要求，没有提供Geometric的众多属性方法，仅是完成在给定size和probs下的分布表示。
3. 三种方式得到的 Geometric Distribution 的表示不同。比如 numpy 使用 ndarray 或者 scalar 进行表示，pytorch 使用 tensor进行表示。
4. pytorch 支持重参数采样，numpy 与 tensorflow 中仅支持sample，如：
```python
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
```

5. numpy 中的 Geometric 仅是一个方法，并不是一个单独的类，因此没有定义属性和其他和前面两者类似的方法。在使用时，只需要指定 Geometric 需要的参数即可得到对应的分布，同时也可以再借助 Matplotlib 来展示 Geometric 分布。

## 方案优点

**pytorch:**
1. 集成的接口比较轻盈，同时实现了 Geometric Distribution 的核心功能。
2. 源码以及实现思路清晰易懂，api 的使用较为方便。

**tensorflow:**
1. 集成的功能众多，几乎囊括了关于 Geometric Distribution 的各个方面，如果使用更复杂的功能，tensorflow可以更加的全面。

**numpy:**
1. numpy 更多是将 Geometric Distribution 当作一种工具，需要使用时可以直接使用，没有再为其封装繁琐的方法和属性。

## 方案缺点
1. tensorflow 因集成的功能众多，所以在使用起来比较麻烦，需要花费更多的时间去学习使用。
2. numpy 则因其缺乏 Geometric Distribution 的其他属性方法，比如 prob / cdf / variance 等，可能会在使用时问题。
3. pytorch 虽然整体简洁轻便，集成了 tensorflow 与 numpy 的各自优势，但是还缺乏一定的额外方法，比如： stddev、copy等。

# 五、设计思路与实现方案

## 命名与参数设计
直接使用 Geometric 分布的名称作为此 API 的名称，参数保持 Geometric 分布最原生的参数即：

- probs：伯努利实验成功概率

预期 paddle 调用 Geometric API 的形式为：
```python
#probs: the probability of sampling 1. Must be in range (0, 1]
paddle.distribution.geometric.Geometric(probs)
```
> 注：Paddle 框架目前只需要支持 probs 所以此次只传递该参数，对于 logits 后期可以尝试针对所有的分布进行同一的支持。
## 底层OP设计

使用 paddle 中现存的 API 进行实现，不考虑再令设计底层 OP。

## API实现方案

该 API 在 `paddle.distribution.geometric` 中实现，部分功能继承父类`Distribution`。
在经过调研对比后，Geometric API 中设定一个参数：probs ，probs 为伯努利实验的成功概率。

除了 API 调用的基本参数外，`paddle.distribution.geometric.Geometric` 中实现的属性、方法主要如下：

### 属性：
- mean ：分布均值
```python
def mean(self):
    return 1. / self.probs - 1.
```

- mode：众数
```python
import paddle

def mode(self):
    return paddle.zeros_like(self.probs)
```
- variance：方差
```python
def variance(self):
    return (1. / self.probs - 1.) / self.probs
```
- stddev：标准差
```python
import math
def stddev(self):
        return math.sqrt(self.variance)
```

### 方法
- sample(shape)：随机采样

```python
import paddle
import numpy as np

def sample(self, shape):
  tiny = np.finfo(self.probs.dtype).tiny
  with paddle.no_grad_():
        u = self.probs.new(shape).uniform_(tiny, 1)
        return (u.log() / (-self.probs).log1p().floor())
```

- prob(value)：概率密度函数

```python
import paddle

def prob(self, value):
    x = paddle.pow(2 * math.pi,-value.shape.pop(1) / 2) * paddle.pow(paddle.linalg.det(self.logits), -1/2)
    y = paddle.exp(-1/2 * paddle.t(value - self.loc) * paddle.inverse(self.logits) * (value - self.loc))
    return x * y
```

- log_prob(value)：对数概率密度函数
```python
def log_prob(self, value):
    value = paddle.broadcast_shape(value, self.probs)
    probs = paddle.broadcast_shape(value, self.probs)
    probs[(probs == 1) & (value ==0)] = 0
    return value * (-probs).log1p() + self.probs.log()
```
- cdf(self): 累计分布函数
```python
def cdf(self):
    n = self.probs.shape.pop(1)
    return 1 - paddle.pow((1 - self.probs), n)
```

- entropy(value)：熵

```python
def entropy(self, value):
    sigma = paddle.linalg.det(self.sclae)
    return 1 / 2 * paddle.log(paddle.pow(2 * math.pi * math.e, value.shpe.pop(1)) * sigma)
```

- kl_divergence 两个Geometric分布之间的kl散度(other--Geometric类的一个实例):
```python
import paddle

def kl_divergence(self, other):
  n = self.probs.shape.pop(1)
  return - n * self.probs * paddle.log(self.probs / other.probs)
```
在`paddle/distribution/kl.py` 中注册`_kl_geometric_geometric`函数，使用时可直接调用`kl_divergence`计算`Geometric`分布之间的kl散度。

# 六、测试和验收的考量

`test_distribution_geometric`继承`unittest.TestCase`类中的方法，参考NormalTest的示例，新增一个`GeometricNumpy`类来验证`Geometric` API的正确性。
- 使用相同的参数实例化 `Geometric` 类和 `GeometricNumpy` 类，分别调用 `mean`、`variance`、`prob`、`log_prob`、`entropy`等方法。将输出的结果进行对比，允许有一定的误差。
- 使用sample，resample方法对多个样本进行测试。

1. 测试 Geometric 分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestGeometric继承unittest.TestCase，分别实现方法setUp（初始化），test_entropy（entropy单测），test_sample（sample单测）。

  * 均值，方差，熵通过Numpy计算相应值，对比Geometric类中相应property的返回值，若一致即正确；

  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合Geometric分布。验证策略如下：随机采样30000个Geometric分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.Geometric`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于geometric分布，若计算所得ks值小于0.1，则拒绝不一致假设，两者属于同一分布；

2. 测试 Geometric 分布的概率密度函数

- 测试方法：该部分主要测试分布各种概率密度函数。类TestGeometricPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测）。

# 七、可行性分析和排期规划

具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.geometric.Geometric` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.geometric.Geometric` API，与飞桨2.4代码风格保持一致

# 名词解释

##伯努利试验
伯努利试验（Bernoulli experiment）是在同样的条件下重复地、相互独立地进行的一种随机试验，其特点是该随机试验只有两种可能结果：发生或者不发生。我们假设该项试验独立重复地进行了n次，那么就称这一系列重复独立的随机试验为n重伯努利试验，或称为伯努利概型。单个伯努利试验是没有多大意义的，然而，当我们反复进行伯努利试验，去观察这些试验有多少是成功的，多少是失败的，事情就变得有意义了，这些累计记录包含了很多潜在的非常有用的信息。

> 参考：百度百科 <伯努利实验>
# 附件及参考资料

## PyTorch

[torch.distributions.geometric.Geometric](https://pytorch.org/docs/stable/distributions.html#geometric)

## TensorFLow

[tfp.distributions.Geometric](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometric#)

## Paddle

[paddle.distribution.Beta](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Beta_cn.html)

## Numpy

[numpy.random.Generator.geometric](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.geometric.html#numpy-random-generator-geometric)
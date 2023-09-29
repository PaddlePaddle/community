# paddle.distribution.binomial 设计文档

|API名称 | paddle.distribution.binomial | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-26 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20230926_api_design_for_binomial.md<br> | 


# 一、概述
## 1、相关背景
提升飞桨 API 丰富度, 需要扩充 API `paddle.distribution.binomial`。

## 2、功能目标
参考 Paddle 现有 distribution，增加 Binomial 分布类的概率统计与随机采样，包括如下方法：
- mean 计算均值
- variance 计算方差
- sample 随机采样
- prob 概率密度
- log_prob 对数概率密度
- entropy 熵计算
- kl_divergence 相对熵计算

## 3、意义
丰富 Paddle 能够提供的分布类型，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了 Distribution 抽象基类，通过继承 Distribution，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 Binomial 概率分布，需要单独开发实现，实现思路与其他概率分布的相同。

# 三、业内方案调研
### Pytorch
PyTorch 中有 API `torch.distributions.binomial.Binomial(total_count=1, probs=None, logits=None, validate_args=None)`
```python
class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """
    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }
    has_enumerate_support = True

    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            (
                self.total_count,
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            (
                self.total_count,
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self):
        return self.total_count * self.probs

    @property
    def mode(self):
        return ((self.total_count + 1) * self.probs).floor().clamp(max=self.total_count)

    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.binomial(
                self.total_count.expand(shape), self.probs.expand(shape)
            )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = (
            self.total_count * _clamp_by_zero(self.logits)
            + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    def entropy(self):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `entropy`."
            )

        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `enumerate_support`."
            )
        values = torch.arange(
            1 + total_count, dtype=self._param.dtype, device=self._param.device
        )
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
```

`torch.distributions.binomial.Binomial`继承自 `torch.distributions.Distribution`

### TensorFlow
TensorFlow 中有 API `tfp.distributions.Binomial(
    total_count,
    logits=None,
    probs=None,
    validate_args=False,
    allow_nan_stats=True,
    name=None
)`

```python
class Binomial(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
  """Binomial distribution.

  This distribution is parameterized by `probs`, a (batch of) probabilities for
  drawing a `1`, and `total_count`, the number of trials per draw from the
  Binomial.

  def __init__(self,
               total_count,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Initialize a batch of Binomial distributions.

    Args:
      total_count: Non-negative floating point tensor with shape broadcastable
        to `[N1,..., Nm]` with `m >= 0` and the same dtype as `probs` or
        `logits`. Defines this as a batch of `N1 x ...  x Nm` different Binomial
        distributions. Its components should be equal to integer values.
      logits: Floating point tensor representing the log-odds of a
        positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
        the same dtype as `total_count`. Each entry represents logits for the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
      probs: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`, `probs in [0, 1]`. Each entry represents the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
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
      raise ValueError(
          'Construct `Binomial` with `probs` or `logits`, but not both.')
    with tf.name_scope(name or 'Binomial') as name:
      dtype = dtype_util.common_dtype([total_count, logits, probs], tf.float32)
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, dtype=dtype, name='total_count')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype=dtype, name='logits')
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype=dtype, name='probs')
      super(Binomial, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        total_count=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED),
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

  @property
  def total_count(self):
    """Number of trials."""
    return self._total_count

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _log_prob(self, counts):
    total_count = tf.convert_to_tensor(self.total_count)
    if self._logits is not None:
      unnorm = _log_unnormalized_prob_logits(self._logits, counts, total_count)
    else:
      unnorm = _log_unnormalized_prob_probs(self._probs, counts, total_count)
    norm = _log_normalization(counts, total_count)
    return unnorm - norm

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _prob(self, counts):
    return tf.exp(self._log_prob(counts))

  def _cdf(self, counts):
    total_count = tf.convert_to_tensor(self.total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    probs, counts = _maybe_broadcast(probs, counts)

    return _bdtr(k=counts, n=total_count, p=probs)

  @distribution_util.AppendDocstring(_binomial_sample_note)
  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed, salt='binomial')
    total_count = tf.convert_to_tensor(self._total_count)
    if self._probs is None:
      probs = self._probs_parameter_no_checks(total_count=total_count)
    else:
      probs = tf.convert_to_tensor(self._probs)

    return _random_binomial(
        shape=ps.convert_to_shape_tensor([n]),
        counts=total_count,
        probs=probs,
        output_dtype=self.dtype,
        seed=seed)[0]

  def _mean(self, probs=None, total_count=None):
    if total_count is None:
      total_count = tf.convert_to_tensor(self._total_count)
    if probs is None:
      probs = self._probs_parameter_no_checks(total_count=total_count)
    return total_count * probs

  def _variance(self):
    total_count = tf.convert_to_tensor(self._total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    return self._mean(probs=probs, total_count=total_count) * (1. - probs)

  @distribution_util.AppendDocstring(
      """Note that when `(1 + total_count) * probs` is an integer, there are
      actually two modes. Namely, `(1 + total_count) * probs` and
      `(1 + total_count) * probs - 1` are both modes. Here we return only the
      larger of the two modes.""")
  def _mode(self):
    total_count = tf.convert_to_tensor(self._total_count)
    probs = self._probs_parameter_no_checks(total_count=total_count)
    return tf.math.minimum(
        total_count, tf.floor((1. + total_count) * probs))

  def logits_parameter(self, name=None):
    """Logits computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      return tf.math.log(probs) - tf.math.log1p(-probs)
    return tensor_util.identity_as_tensor(self._logits)

  def probs_parameter(self, name=None):
    """Probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self, total_count=None):
    if self._logits is None:
      probs = tensor_util.identity_as_tensor(self._probs)
    else:
      probs = tf.math.sigmoid(self._logits)
    # Suppress potentially nasty probs like `nan` b/c they don't matter where
    # total_count == 0.
    if total_count is None:
      total_count = self.total_count
    return tf.where(total_count > 0, probs, 0)

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    assertions = []

    if is_init != tensor_util.is_ref(self.total_count):
      total_count = tf.convert_to_tensor(self.total_count)
      msg1 = 'Argument `total_count` must be non-negative.'
      msg2 = 'Argument `total_count` cannot contain fractional components.'
      assertions += [
          assert_util.assert_non_negative(total_count, message=msg1),
          distribution_util.assert_integer_form(total_count, message=msg2),
      ]

    if self._probs is not None:
      if is_init != tensor_util.is_ref(self._probs):
        probs = tf.convert_to_tensor(self._probs)
        one = tf.constant(1., probs.dtype)
        assertions += [
            assert_util.assert_non_negative(
                probs, message='probs has components less than 0.'),
            assert_util.assert_less_equal(
                probs, one, message='probs has components greater than 1.')
        ]

    return assertions

  def _sample_control_dependencies(self, counts):
    """Check counts for proper values."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(distribution_util.assert_casting_closed(
        counts, target_dtype=tf.int32,
        message='counts cannot contain fractional components.'))
    assertions.append(assert_util.assert_non_negative(
        counts, message='counts must be non-negative.'))
    assertions.append(
        assert_util.assert_less_equal(
            counts, self.total_count,
            message=('Sampled counts must be itemwise less than '
                     'or equal to `total_count` parameter.')))
    return assertions
```

`tfp.distributions.Binomial` 继承自 `tfp.distribution.DiscreteDistributionMixin` 和 `tfp.distribution.AutoCompositeTensorDistribution`

# 四、对比分析
Pytorch 与 Tensorflow 实现方式大体类似，都是通过基本的概率计算得到相应的概率属性。Tensorflow 实现的方法更丰富，测试更为详细。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.binomial(total_count, probs)
```
参数 `total_count`, `probs` 分别为 Binomial 分布的两个参数。

例如，随机变量 $X$ 服从 Binomial 分布，即 $X \sim Binomial(n, p)$ ，对应的参数 `total_count`$=n$ ， `probs`$=p$ 。

## 底层OP设计
本次任务的设计思路与已有概率分布保持一致，不涉及底层 OP 的开发。

## API实现方案
新增 `Binomial` 类

```python
class Binomial(Distribution):
  def __init__(self, total_count, probs):
    super().__init__(batch_shape=self.probs.shape, event_shape=())
    
    ...
    
```

`Binomial` 类的初始化参数有 `total_count` 和 `probs` ，类包含的方法及实现方案如下：

记参数 `total_count`$=n$ ， `probs`$=p$ 。

- `mean` 计算均值

均值的计算方法： $n p$

- `variance` 计算方差

方差的计算方法： $n p (1 - p)$

- `entropy` 熵计算

熵的计算方法： $H = - \sum_x f(x) \log{f(x)}$

- `kl_divergence` 相对熵计算

相对熵的计算方法： $D_{KL}(n_1, p_1, n_2, p_2) = \sum_x f_1(x) \log{\frac{f_1(x)}{f_2(x)}}$

- `sample` 随机采样

采样方法： 可以利用 `paddle.distribution.bernoulli.sample()`进行采样

- `prob` 概率密度

概率密度计算方法： $f(x;n,p) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}$

- `log_prob` 对数概率密度

对数概率密度计算方法： $\log[f(x;n,p)] = \log[\frac{n!}{x!(n-x)!}] + x \log p + (n-x) \log (1-p)$



# 六、测试和验收的考量
`Binomial` 类测试以 Numpy 作为基准，验证API的正确性。
1. 使用 Numpy 实现所有 Binomial 的API，集成为 `BinomialNumpy` 类，用以验证本次任务开发的 API 的正确性。

2. 使用同样的参数实例化 `Binomial` 类和 `BinomialNumpy` 类，并调用 `mean`、`variance`、`entropy`、`prob`、`kl_divergence`方法，测试结果是否相等（容许一定误差）。参数 `total_count` 和 `probs` 的支持的数据类型需测试详尽。

3. 使用 `Binomial` 类的 `sample` 方法生成5000个样本，测试这些这样的均值和标准差是否正确。


# 七、可行性分析和排期规划
- 排期规划

9月27日~10月4日完成API开发与调试。

10月5日~10月12日完成测试代码的开发。

# 八、影响面
本次任务影响的模块如下：
1. `paddle.distribution` 

新增 binomial.py 文件。

2. `./test/distribution`

新增 test_distribution_binomial.py 和 test_distribution_binomial_static.py 文件。

# 名词解释
- Binomial 分布

若随机变量 $X \sim Binomial(n, p)$，则 $X$ 的概率密度函数为
$$f(x;n,p) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}$$

# 附件及参考资料
1. [Tensorflow 的 Binomial 文档](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Binomial)

2. [Pytorch 的 Binomial 文档](https://pytorch.org/docs/stable/distributions.html#binomial)

3. [Numpy 的 Binomial 文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)

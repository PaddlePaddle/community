# paddle.distribution.continuous_bernoulli 设计文档

|API名称 | paddle.distribution.continuous_bernoulli | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-27 | 
|版本号 | V1.2 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20230927_api_design_for_continuous_bernoulli.md<br> | 


# 一、概述
## 1、相关背景
连续伯努利分布是一种定义在 $[0, 1]$ 闭区间上的概率分布, 它具有一个描述分布函数的形状的参数 $\lambda \in (0, 1)$, 该参数对连续伯努利概率密度函数的影响如下: $f(x|\lambda) \propto \lambda^x (1 - \lambda)^{1-x}$。它属于指数分布族, 可以被看作连续版的伯努利分布。而在机器学习中, 伯努利分布可以推导出二元数据的交叉熵损失, 于是连续伯努利分布就可以为连续型数据提供一种类似交叉熵的损失衡量方法。例如在 VAE 中, 若数据非二元分布而是分布在 $[0, 1]$ 区间上, [The continuous Bernoulli: fixing a pervasive error in
variational autoencoders](https://proceedings.neurips.cc/paper_files/paper/2019/file/f82798ec8909d23e55679ee26bb26437-Paper.pdf)指出: 如果用传统做法先将其转化为二元 $\{0, 1\}$ 分布, 再将先验分布选择为普通的伯努利分布, 这样的做法是不能够很好的用统计理论来进行解释的, 因为在非监督学习中为了fit模型去修改数据这样的做法本身就有问题, 会丢失很多信息等等。 于是这篇论文提出了这种新的分布作为 VAE 的先验分布用以处理连续型数据。因此为提升飞桨的概率分布 API 丰富度, 从而为更多的机器学习应用提供可能, 需要扩充 API `paddle.distribution.continuous_bernoulli`。

## 2、功能目标
参考 Paddle 现有 distribution，增加 ContinuousBernoulli 分布类的概率统计与随机采样，包括如下方法：
- mean 计算均值
- variance 计算方差
- sample 随机采样
- rsample 重参数化随机采样
- prob 概率密度
- log_prob 对数概率密度
- cdf 累积概率密度
- icdf 累积概率密度逆函数
- entropy 熵计算
- kl_divergence 相对熵计算

## 3、意义
丰富 Paddle 能够提供的分布类型，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了 Distribution 抽象基类，通过继承 Distribution，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 ContinuousBernoulli 概率分布，需要单独开发实现，实现思路与其他概率分布的相同。

# 三、业内方案调研
### Pytorch
PyTorch 中有 API `torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=None, logits=None, lims=(0.499, 0.501), validate_args=None)`
```python
class ContinuousBernoulli(ExponentialFamily):
    r"""
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by 'probs' (in
    (0,1)) or 'logits' (real-valued). Note that, unlike the Bernoulli, 'probs'
    does not correspond to a probability and 'logits' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    _mean_carrier_measure = 0
    has_rsample = True

    def __init__(
        self, probs=None, logits=None, lims=(0.499, 0.501), validate_args=None
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            (self.probs,) = broadcast_all(probs)
            # validate 'probs' here if necessary as it is later clamped for numerical stability
            # close to 0 and 1, later on; otherwise the clamped 'probs' would always pass
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            self.probs = clamp_probs(self.probs)
        else:
            is_scalar = isinstance(logits, Number)
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        self._lims = lims
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        new._lims = self._lims
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def _outside_unstable_region(self):
        return torch.max(
            torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1])
        )

    def _cut_probs(self):
        return torch.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * torch.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(
            torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs)
        )
        cut_probs_above_half = torch.where(
            torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs)
        )
        log_norm = torch.log(
            torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))
        ) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    @property
    def mean(self):
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            torch.log1p(-cut_probs) - torch.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * torch.pow(x, 2)) * x
        return torch.where(self._outside_unstable_region(), mus, taylor)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / torch.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / torch.pow(torch.log1p(-cut_probs) - torch.log(cut_probs), 2)
        x = torch.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return torch.where(self._outside_unstable_region(), vars, taylor)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    @property
    def param_shape(self):
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with torch.no_grad():
            return self.icdf(u)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cut_probs = self._cut_probs()
        cdfs = (
            torch.pow(cut_probs, value) * torch.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = torch.where(self._outside_unstable_region(), cdfs, value)
        return torch.where(
            torch.le(value, 0.0),
            torch.zeros_like(value),
            torch.where(torch.ge(value, 1.0), torch.ones_like(value), unbounded_cdfs),
        )

    def icdf(self, value):
        cut_probs = self._cut_probs()
        return torch.where(
            self._outside_unstable_region(),
            (
                torch.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - torch.log1p(-cut_probs)
            )
            / (torch.log(cut_probs) - torch.log1p(-cut_probs)),
            value,
        )

    def entropy(self):
        log_probs0 = torch.log1p(-self.probs)
        log_probs1 = torch.log(self.probs)
        return (
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    @property
    def _natural_params(self):
        return (self.logits,)

    def _log_normalizer(self, x):
        """computes the log normalizing constant as a function of the natural parameter"""
        out_unst_reg = torch.max(
            torch.le(x, self._lims[0] - 0.5), torch.gt(x, self._lims[1] - 0.5)
        )
        cut_nat_params = torch.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * torch.ones_like(x)
        )
        log_norm = torch.log(torch.abs(torch.exp(cut_nat_params) - 1.0)) - torch.log(
            torch.abs(cut_nat_params)
        )
        taylor = 0.5 * x + torch.pow(x, 2) / 24.0 - torch.pow(x, 4) / 2880.0
        return torch.where(out_unst_reg, log_norm, taylor)
```

`torch.distributions.continuous_bernoulli.ContinuousBernoulli`继承自 `torch.distributions.ExponentialFamily`

### TensorFlow
TensorFlow 中有 API `tfp.distributions.ContinuousBernoulli(
      logits=None,
      probs=None,
      dtype=tf.float32,
      validate_args=False,
      allow_nan_stats=True,
      name='ContinuousBernoulli'
)`

```python
class ContinuousBernoulli(distribution.AutoCompositeTensorDistribution):

  def __init__(
      self,
      logits=None,
      probs=None,
      dtype=tf.float32,
      validate_args=False,
      allow_nan_stats=True,
      name='ContinuousBernoulli'):
    """Construct Bernoulli distributions.

    Args:
      logits: An N-D `Tensor`. Each entry in the `Tensor` parameterizes
       an independent continuous Bernoulli distribution with parameter
       sigmoid(logits). Only one of `logits` or `probs` should be passed
       in. Note that this does not correspond to the log-odds as in the
       Bernoulli case.
      probs: An N-D `Tensor` representing the parameter of a continuous
       Bernoulli. Each entry in the `Tensor` parameterizes an independent
       continuous Bernoulli distribution. Only one of `logits` or `probs`
       should be passed in. Note that this also does not correspond to a
       probability as in the Bernoulli case.
      dtype: The type of the event samples. Default: `float32`.
       validate_args: Python `bool`, default `False`. When `True`
       distribution parameters are checked for validity despite possibly
       degrading runtime performance. When `False` invalid inputs may
       silently render incorrect outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is
        raised if one or more of the statistic's batch members are
        undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: If probs and logits are passed, or if neither are passed.
    """
    parameters = dict(locals())
    if (probs is None) == (logits is None):
      raise ValueError('Must pass `probs` or `logits`, but not both.')
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=tf.float32, name='logits')
    super(ContinuousBernoulli, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        logits=parameter_properties.ParameterProperties(),
        probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False))

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

  def _sample_n(self, n, seed=None):
    logits = self._logits_parameter_no_checks()
    new_shape = ps.concat([[n], ps.shape(logits)], axis=0)
    uniform = samplers.uniform(new_shape, seed=seed, dtype=logits.dtype)
    sample = self._quantile(uniform, logits)
    return tf.cast(sample, self.dtype)

  def _log_normalizer(self, logits=None):
    # The normalizer is 2 * atanh(1 - 2 * probs) / (1 - 2 * probs), with the
    # removable singularity at probs = 0.5 removed (and replaced with 2).
    # We do this computation in logit space to be more numerically stable.
    # Note that 2 * atanh(1 - 2 / (1 + exp(-logits))) = -logits.
    # Thus we end up with
    # -logits / (1 - 2 / (1 + exp(-logits))) =
    # logits / ((-exp(-logits) + 1) / (exp(-logits) + 1)) =
    # (exp(-logits) + 1) * logits / (-exp(-logits) + 1) =
    # (1 + exp(logits)) * logits / (exp(logits) - 1)

    if logits is None:
      logits = self._logits_parameter_no_checks()
    return _log_xexp_ratio(logits)

  def _log_prob(self, event):
    log_probs0, log_probs1, _ = self._outcome_log_probs()
    event = tf.cast(event, log_probs0.dtype)
    tentative_log_pdf = (tf.math.multiply_no_nan(log_probs0, 1.0 - event)
                         + tf.math.multiply_no_nan(log_probs1, event)
                         + self._log_normalizer())
    return tf.where(
        (event < 0) | (event > 1),
        dtype_util.as_numpy_dtype(log_probs0.dtype)(-np.inf),
        tentative_log_pdf)

  def _log_cdf(self, x):
    # The CDF is (p**x * (1 - p)**(1 - x) + p - 1) / (2 * p - 1).
    # We do this computation in logit space to be more numerically stable.
    # p**x * (1- p)**(1 - x) becomes
    # 1 / (1 + exp(-logits))**x *
    # exp(-logits * (1 - x)) / (1 + exp(-logits)) ** (1 - x) =
    # exp(-logits * (1 - x)) / (1 + exp(-logits))
    # p - 1 becomes -exp(-logits) / (1 + exp(-logits))
    # Thus the whole numerator is
    # (exp(-logits * (1 - x)) - exp(-logits)) / (1 + exp(-logits))
    # The denominator is (1 - exp(-logits)) / (1 + exp(-logits))
    # Putting it all together, this gives:
    # (exp(-logits * (1 - x)) - exp(-logits)) / (1 - exp(-logits)) =
    # (exp(logits * x) - 1) / (exp(logits) - 1)
    logits = self._logits_parameter_no_checks()

    # For logits < 0, we can directly use the expression.
    safe_logits = tf.where(logits < 0., logits, -1.)
    result_negative_logits = (
        generic.log1mexp(
            tf.math.multiply_no_nan(safe_logits, x)) -
        generic.log1mexp(safe_logits))
    # For logits > 0, to avoid infs with large arguments we rewrite the
    # expression. Let z = log(exp(logits) - 1)
    # log_cdf = log((exp(logits * x) - 1) / (exp(logits) - 1))
    #         = log(exp(logits * x) - 1) - log(exp(logits) - 1)
    #         = log(exp(logits * x) - 1) - log(exp(z))
    #         = log(exp(logits * x - z) - exp(-z))
    # Because logits > 0, logits * x - z > -z, so we can pull it out to get
    #         = log(exp(logits * x - z) * (1 - exp(-logits * x)))
    #         = logits * x - z + tf.math.log(1 - exp(-logits * x))
    dtype = dtype_util.as_numpy_dtype(x.dtype)
    eps = np.finfo(dtype).eps
    # log(exp(logits) - 1)
    safe_logits = tf.where(logits > 0., logits, 1.)
    z = tf.where(
        safe_logits > -np.log(eps),
        safe_logits, tf.math.log(tf.math.expm1(safe_logits)))
    result_positive_logits = tf.math.multiply_no_nan(
        safe_logits, x) - z + generic.log1mexp(
            -tf.math.multiply_no_nan(safe_logits, x))

    result = tf.where(
        logits < 0., result_negative_logits,
        tf.where(logits > 0., result_positive_logits, tf.math.log(x)))

    # Finally, handle the case where `logits` and `p` are on the boundary,
    # as the above expressions can result in ratio of `infs` in that case as
    # well.
    result = tf.where(
        tf.math.equal(logits, np.inf), dtype(-np.inf), result)
    result = tf.where(
        (tf.math.equal(logits, -np.inf) & tf.math.not_equal(x, 0.)) | (
            tf.math.equal(logits, np.inf) & tf.math.equal(x, 1.)),
        tf.zeros_like(logits), result)

    result = tf.where(
        x < 0.,
        dtype(-np.inf),
        tf.where(x > 1., tf.zeros_like(x), result))

    return result

  def _outcome_log_probs(self):
    """Returns log(1-probs), log(probs) and logits."""
    if self._logits is None:
      probs = tf.convert_to_tensor(self._probs)
      logits = tf.math.log(probs) - tf.math.log1p(-probs)
      return tf.math.log1p(-probs), tf.math.log(probs), logits
    s = tf.convert_to_tensor(self._logits)
    # softplus(s) = -Log[1 - p]
    # -softplus(-s) = Log[p]
    # softplus(+inf) = +inf, softplus(-inf) = 0, so...
    #  logits = -inf ==> log_probs0 = 0, log_probs1 = -inf (as desired)
    #  logits = +inf ==> log_probs0 = -inf, log_probs1 = 0 (as desired)
    return -tf.math.softplus(s), -tf.math.softplus(-s), s

  def _entropy(self):
    log_probs0, log_probs1, logits = self._outcome_log_probs()
    return (self._mean(logits) * (log_probs0 - log_probs1)
            - self._log_normalizer(logits) - log_probs0)

  def _mean(self, logits=None):
    # The mean is probs / (2 * probs - 1) + 1 / (2 * arctanh(1 - 2 * probs))
    # with the removable singularity at 0.5 removed.
    # We write this in logits space.
    # The first term becomes
    # 1 / (1 + exp(-logits)) / (2 / (1 + exp(-logits)) - 1) =
    # 1 / (2 - 1 - exp(-logits)) =
    # 1 / (1 - exp(-logits))
    # The second term becomes - 1 / logits.
    # Thus we have mean = 1 / (1 - exp(-logits)) - 1 / logits.

    # When logits is close to zero, we can compute the Laurent series for the
    # first term as:
    # 1 / x + 1 / 2 + x / 12 - x**3 / 720 + x**5 / 30240 + O(x**7).
    # Thus we get the pole at zero canceling out with the second term.

    # For large negative logits, the denominator (1 - exp(-logits)) in
    # the first term yields inf values. Whilst the ratio still returns
    # zero as it should, the gradients of this ratio become nan.
    # Thus, noting that 1 / (1 - exp(-logits)) quickly tends towards 0
    # for large negative logits, the mean tends towards - 1 / logits.

    dtype = dtype_util.as_numpy_dtype(self.dtype)
    eps = np.finfo(dtype).eps

    if logits is None:
      logits = self._logits_parameter_no_checks()

    small_cutoff = np.power(eps * 30240, 1 / 5.)
    result = dtype(0.5) + logits / 12. - logits * tf.math.square(logits) / 720

    large_cutoff = -np.log(eps)

    safe_logits_mask = ((tf.math.abs(logits) > small_cutoff)
                        & (logits > -large_cutoff))
    safe_logits = tf.where(safe_logits_mask, logits, dtype(1.))
    result = tf.where(
                safe_logits_mask,
                -(tf.math.reciprocal(
                    tf.math.expm1(-safe_logits)) +
                  tf.math.reciprocal(safe_logits)),
                result)

    large_neg_mask = logits <= -large_cutoff
    logits_large_neg = tf.where(large_neg_mask, logits, 1.)
    return tf.where(large_neg_mask,
                    -tf.math.reciprocal(logits_large_neg),
                    result)

  def _variance(self):
    # The variance is var = probs (probs - 1) / (2 * probs - 1)**2 +
    # 1 / (2 * arctanh(1 - 2 * probs))**2
    # with the removable singularity at 0.5 removed.
    # We write this in logits space.
    # Let v = 1 + exp(-logits) = 1 / probs
    # The first term becomes
    # probs * (probs - 1) / (2 * probs - 1)**2 turns in to:
    # 1 / v * (1 / v - 1) / (2 / v - 1) ** 2 =
    # (1 / v ** 2) * (1 - v) / (2 / v - 1) ** 2 =
    # (1 - v) / (2 - v) ** 2 =
    # -exp(-logits) / (1 - exp(-logits))**2 =
    # -exp(-logits) / (1 + exp(-2 * logits) - 2 * exp(-logits)) =
    # -1 / (exp(logits) + exp(-logits) - 2) =
    # 1 / (2 - 2 * cosh(logits))
    # For the second term, we have
    # 1 / (2 * arctanh(1 - 2 * probs))**2 =
    # 1 / (2 * 0.5 * log((1 + 1 - 2 * probs) / (1 - (1 - 2 * probs))))**2 =
    # 1 / (log(2 * (1 - probs) / (2 * probs)))**2 =
    # 1 / (log((1 - probs) / probs))**2 =
    # 1 / (log(1 / probs - 1))**2 =
    # 1 / (log(1 + exp(-logits) - 1))**2 =
    # 1 / (-logits)**2 =
    # 1 / logits**2

    # Thus we have var = 1 / (2 - 2 * cosh(logits)) + 1 / logits**2

    # For the function f(x) = exp(-x) / (1 - exp(-x)) ** 2 + 1 / x ** 2, when
    # logits is close to zero, we can compute the Laurent series for the first
    # term as:
    # -1 / x**2 + 1 / 12 - x**2 / 240 + x**4 / 6048 + x**6 / 172800 + O(x**8).
    # Thus we get the pole at zero canceling out with the second term.

    dtype = dtype_util.as_numpy_dtype(self.dtype)
    eps = np.finfo(dtype).eps

    logits = self._logits_parameter_no_checks()

    small_cutoff = np.power(eps * 172800, 1 / 6.)
    logits_sq = tf.math.square(logits)
    small_result = (dtype(1 / 12.) - logits_sq / 240. +
                    tf.math.square(logits_sq) / 6048)

    safe_logits_large = tf.where(
        tf.math.abs(logits) > small_cutoff, logits, dtype(1.))
    return tf.where(
        tf.math.abs(logits) > small_cutoff,
        (tf.math.reciprocal(2 * (1. - tf.math.cosh(safe_logits_large))) +
         tf.math.reciprocal(tf.math.square(safe_logits_large))),
        small_result)

  def _quantile(self, p, logits=None):
    if logits is None:
      logits = self._logits_parameter_no_checks()
    logp = tf.math.log(p)
    # The expression for the quantile function is:
    # log(1 + (e^s - 1) * p) / s, where s is `logits`. When s is large,
    # the e^s sub-term becomes increasingly ill-conditioned.  However,
    # since the numerator tends to s, we can reformulate the s > 0 case
    # as a offset from 1, which is more accurate.  Coincidentally,
    # this eliminates a ratio of infinities problem when `s == +inf`.

    safe_negative_logits = tf.where(logits < 0., logits, -1.)
    safe_positive_logits = tf.where(logits > 0., logits, 1.)
    result = tf.where(
        logits > 0.,
        1. + generic.log_add_exp(
            logp + generic.log1mexp(safe_positive_logits),
            tf.math.negative(safe_positive_logits)) / safe_positive_logits,
        tf.math.log1p(
            tf.math.expm1(safe_negative_logits) * p) / safe_negative_logits)

    # When logits is zero, we can simplify
    # log(1 + (e^s - 1) * p) / s ~= log(1 + s * p) / s ~= s * p / s = p
    # Specifically, when logits is zero, the naive computation produces a NaN.
    result = tf.where(tf.math.equal(logits, 0.), p, result)

    # Finally, handle the case where `logits` and `p` are on the boundary,
    # as the above expressions can result in ratio of `infs` in that case as
    # well.
    return tf.where(
        (tf.math.equal(logits, -np.inf) & tf.math.equal(logp, 0.)) |
        (tf.math.equal(logits, np.inf) & tf.math.is_inf(logp)),
        tf.ones_like(logits),
        result)

  def _mode(self):
    """Returns `1` if `prob > 0.5` and `0` otherwise."""
    return tf.cast(self._probs_parameter_no_checks() > 0.5, self.dtype)

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
    """probs computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.sigmoid(self._logits)

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_continuous_bernoulli_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))
    assertions.append(
        assert_util.assert_less_equal(
            x,
            tf.ones([], dtype=x.dtype),
            message='Sample must be less than or equal to `1`.'))
    return assertions
```

`tfp.distributions.ContinuousBernoulli` 继承自 `tfp.distribution.AutoCompositeTensorDistribution`

# 四、对比分析
由于连续伯努利的分布函数以及密度函数都有显示表达式，Pytorch 与 Tensorflow 对其实现方式大体类似，都是通过基本的概率计算得到相应的概率属性。对于 Exponential Family , Pytorch 设计的 `ExponentialFamily` 类主要用于通过[Bregman Divergence](https://www.researchgate.net/publication/221126408_Entropies_and_cross-entropies_of_exponential_families)这种统一的方法来计算指数族分布的 entropy 和 kl_divergence , 但继承之后也需要根据理论计算写出每个指数族分布对应的 `natural parameters` 以及其 `log normalizer`, 和 `mean carrier measure` , 通过这三个方法来计算 entropy 和 kl_divergence 。Poisson 分布的 entropy 和 kl_divergence 的无论是直接按定义计算还是用 Bregman Divergence 来计算都没有显式的表达式, 因为涉及到 $[0, \infty) \cap \mathbb{N}$ 的支撑集, Pytorch 中 Poisson 的 `entropy` 也没有实现完毕( `mean carrier measure` 未实现), 所以此处建议还是先继承基础的 `Distribution` , `entropy` 和 `kl_divergence` 的实现按第五部分描述的方法直接将进行计算。另外，由于概率分布的数值计算在 `probs` = 0.5 附近不稳定(根据表达式0.5是一个奇点), 而 Pytorch 与 Tensorflow 的处理方式类似, Pytorch 基于 probs 来计算, 通过传入参数 `lims` 让用户可以自己定义不稳定区域的范围, 在不稳定区域内利用泰勒展开做近似计算, Tensorflow 则是基于 logits 来计算, 而 Tensorflow 对不稳定区域的范围无法让用户自定义。 此外 Pytorch 的计算代码更加直观简洁。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.continuous_bernoulli(probability, eps=0.02)
```
- 参数 `probability` 为 ContinuousBernoulli 分布的参数。
- 参数 `eps` 表示非稳定计算区域的带宽，非稳定计算区域为 $[0.5 - eps, 0.5 + eps]$

例如，随机变量 $X$ 服从 ContinuousBernoulli 分布，即 $X \sim ContinuousBernoulli(\lambda)$ ，对应的参数 `probs`$=\lambda$。

## 底层OP设计
本次任务的设计思路与已有概率分布保持一致，不涉及底层 OP 的开发。

## API实现方案
新增 `ContinuousBernoulli` 类

```python
class ContinuousBernoulli(Distribution):
  def __init__(self, probability, eps=0.02):
    super().__init__(batch_shape=self.probability.shape, event_shape=())
    
    ...
    
```

`ContinuousBernoulli` 类的初始化参数是 `probability` 和 `eps` ，类包含的方法及实现方案如下：

记参数 `probs`$=\lambda$, `eps` 为非稳定计算区域范围: $[0.5-eps, 0.5+eps]$

- `mean` 计算均值

均值的计算方法： 
```math
E(X) = 
\left\{ 
\begin{aligned}
&\frac{1}{2} & \text{ if $\lambda = \frac{1}{2}$}\\ 
&\frac{\lambda}{2\lambda - 1} + \frac{1}{2\tanh^{-1}(1-2\lambda)} & \text{ otherwise}
\end{aligned}
\right.
```

- `variance` 计算方差

方差的计算方法： 
```math
Var(X) = 
\left\{ 
\begin{aligned}
&\frac{1}{12} & \text{ if $\lambda = \frac{1}{2}$} \\ 
&\frac{-(1 - \lambda)\lambda}{(1 - 2\lambda)^2} + \frac{1}{(2\tanh^{-1}(1-2\lambda))^2}   & \text{ otherwise}
\end{aligned}
\right.
```

- `entropy` 熵计算

熵的计算方法： $H = - \sum_x f(x) \log{f(x)}$

- `kl_divergence` 相对熵计算

相对熵的计算方法： $D_{KL}(\lambda_1, \lambda_2) = \sum_x f_1(x) \log{\frac{f_1(x)}{f_2(x)}}$

- `sample` 随机采样

采样方法： 利用`icdf`进行采样

- `rsample` 随机采样

采样方法： 利用`icdf`进行采样

- `prob` 概率密度

概率密度计算方法： $$f(x;\lambda) = C(\lambda)\lambda^x (1-\lambda)^{1-x}$$
```math
C(\lambda) = 
\left\{ 
\begin{aligned}
&2 & \text{ if $\lambda = \frac{1}{2}$} \\ 
&\frac{2\tanh^{-1}(1-2\lambda)}{1 - 2\lambda} & \text{ otherwise}
\end{aligned}
\right.
```

- `log_prob` 对数概率密度

对数概率密度计算方法： 概率密度取对数

- `cdf` 累积概率密度

累积概率密度计算方法： 
```math
P(X \le t; \lambda) = 
F(t;\lambda) = 
\left\{ 
\begin{aligned}
&t & \text{ if $\lambda = \frac{1}{2}$} \\ 
&\frac{\lambda^t (1 - \lambda)^{1 - t} + \lambda - 1}{2\lambda - 1} & \text{ otherwise}
\end{aligned}
\right.
```

- `icdf` 累积概率密度逆函数
```math
F^{-1}(x;\lambda) = 
\left\{ 
\begin{aligned}
&x & \text{ if $\lambda = \frac{1}{2}$} \\ 
&\frac{\log(1+(\frac{2\lambda - 1}{1 - \lambda})x)}{\log(\frac{\lambda}{1-\lambda})} & \text{ otherwise}
\end{aligned}
\right.
```


# 六、测试和验收的考量
`ContinuousBernoulli` 类测试以 Numpy 作为基准，用 Numpy 实现所有 API , 以验证正确性。

1. 使用同样的参数实例化 `ContinuousBernoulli` 类和 `ContinuousBernoulli_np` 类，并调用 `mean`、`variance`、`entropy`、`log_prob`、`cdf`、`icdf`、`kl_divergence`等方法，测试结果是否相等（容许一定误差）。

2. 使用 `ContinuousBernoulli` 类的 `sample` 方法生成5000个样本，测试这些这样的均值和标准差是否正确。


# 七、可行性分析和排期规划
- 排期规划

10月13日~10月20日完成API开发与调试。

10月21日~10月28日完成测试代码的开发。

# 八、影响面
本次任务影响的模块如下：
1. `paddle.distribution` 

新增 continuous_bernoulli.py 文件。

2. `./test/distribution`

新增 test_distribution_continuous_bernoulli.py 和 test_distribution_continuous_bernoulli_static.py 文件。

# 名词解释
- ContinuousBernoulli 分布

若随机变量 $X \sim ContinuousBernoulli(\lambda)$，则 $X$ 的概率密度函数为
$$f(x;\lambda) = C(\lambda)\lambda^x (1-\lambda)^{1-x}$$
```math
C(\lambda) = 
\left\{ 
\begin{aligned}
&2 & \text{ if $\lambda = \frac{1}{2}$} \\ 
&\frac{2\tanh^{-1}(1-2\lambda)}{1 - 2\lambda} & \text{ otherwise}
\end{aligned}
\right.
```

# 附件及参考资料
1. [Pytorch 的 ContinuousBernoulli 文档](https://pytorch.org/docs/stable/distributions.html#continuousbernoulli)

2. [Wiki 的 ContinuousBernoulli 介绍](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution)

# paddle.distribution.poisson 设计文档

|API名称 | paddle.distribution.poisson | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-26 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20230926_api_design_for_poisson.md<br> | 


# 一、概述
## 1、相关背景
提升飞桨 API 丰富度, 需要扩充 API `paddle.distribution.poisson`。

## 2、功能目标
参考 Paddle 现有 distribution，增加 Poisson 分布类的概率统计与随机采样，包括如下方法：
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
Paddle 框架内定义了 Distribution 抽象基类，通过继承 Distribution，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 Poisson 概率分布，需要单独开发实现，实现思路与其他概率分布的相同。

# 三、业内方案调研
### Pytorch
PyTorch 中有 API `torch.distributions.poisson.Poisson(rate, validate_args=None)`
```python
class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """
    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate

    @property
    def mode(self):
        return self.rate.floor()

    @property
    def variance(self):
        return self.rate

    def __init__(self, rate, validate_args=None):
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        return value.xlogy(rate) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self):
        return (torch.log(self.rate),)

    def _log_normalizer(self, x):
        return torch.exp(x)
```

`torch.distributions.poisson.Poisson`继承自 `torch.distributions.ExponentialFamily`

### TensorFlow
TensorFlow 中有 API `tfp.distributions.Poisson(
    rate=None,
    log_rate=None,
    force_probs_to_zero_outside_support=False,
    validate_args=False,
    allow_nan_stats=True,
    name='Poisson'
)`

```python
class Poisson(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):

  def __init__(self,
               rate=None,
               log_rate=None,
               force_probs_to_zero_outside_support=False,
               validate_args=False,
               allow_nan_stats=True,
               name='Poisson'):

    parameters = dict(locals())
    if (rate is None) == (log_rate is None):
      raise ValueError('Must specify exactly one of `rate` and `log_rate`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([rate, log_rate], dtype_hint=tf.float32)
      if not dtype_util.is_floating(dtype):
        raise TypeError('[log_]rate.dtype ({}) is a not a float-type.'.format(
            dtype_util.name(dtype)))
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, name='rate', dtype=dtype)
      self._log_rate = tensor_util.convert_nonref_to_tensor(
          log_rate, name='log_rate', dtype=dtype)
      self._force_probs_to_zero_outside_support = force_probs_to_zero_outside_support

      super(Poisson, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        rate=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False),
        log_rate=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log rate parameter."""
    return self._log_rate

  @property
  def force_probs_to_zero_outside_support(self):
    """Return 0 probabilities on non-integer inputs."""
    return self._force_probs_to_zero_outside_support

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _log_prob(self, x):
    log_rate = self._log_rate_parameter_no_checks()
    log_probs = (self._log_unnormalized_prob(x, log_rate) -
                 self._log_normalization(log_rate))
    if self.force_probs_to_zero_outside_support:
      # Ensure the gradient wrt `rate` is zero at non-integer points.
      log_probs = tf.where(
          tf.math.is_inf(log_probs),
          dtype_util.as_numpy_dtype(log_probs.dtype)(-np.inf),
          log_probs)
    return log_probs

  def _log_cdf(self, x):
    return tf.math.log(self.cdf(x))

  def _cdf(self, x):
    # CDF is the probability that the Poisson variable is less or equal to x.
    # For fractional x, the CDF is equal to the CDF at n = floor(x).
    # For negative x, the CDF is zero, but tf.igammac gives NaNs, so we impute
    # the values and handle this case explicitly.
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    cdf = tf.math.igammac(1. + safe_x, self._rate_parameter_no_checks())
    return tf.where(x < 0., tf.zeros_like(cdf), cdf)

  def _log_survival_function(self, x):
    return tf.math.log(self.survival_function(x))

  def _survival_function(self, x):
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    survival = tf.math.igamma(1. + safe_x, self._rate_parameter_no_checks())
    return tf.where(x < 0., tf.ones_like(survival), survival)

  def _log_normalization(self, log_rate):
    return tf.exp(log_rate)

  def _log_unnormalized_prob(self, x, log_rate):
    # The log-probability at negative points is always -inf.
    # Catch such x's and set the output value accordingly.
    safe_x = tf.maximum(
        tf.floor(x) if self.force_probs_to_zero_outside_support else x, 0.)
    y = tf.math.multiply_no_nan(log_rate, safe_x) - tf.math.lgamma(1. + safe_x)
    return tf.where(
        tf.equal(x, safe_x), y, dtype_util.as_numpy_dtype(y.dtype)(-np.inf))

  def _mean(self):
    return self._rate_parameter_no_checks()

  def _variance(self):
    return self._rate_parameter_no_checks()

  @distribution_util.AppendDocstring(
      """Note: when `rate` is an integer, there are actually two modes: `rate`
      and `rate - 1`. In this case we return the larger, i.e., `rate`.""")
  def _mode(self):
    return tf.floor(self._rate_parameter_no_checks())

  def _sample_n(self, n, seed=None):
    seed = samplers.sanitize_seed(seed)
    return random_poisson(
        shape=ps.convert_to_shape_tensor([n]),
        rates=(None if self._rate is None else
               tf.convert_to_tensor(self._rate)),
        log_rates=(None if self._log_rate is None else
                   tf.convert_to_tensor(self._log_rate)),
        output_dtype=self.dtype,
        seed=seed)[0]

  def rate_parameter(self, name=None):
    """Rate vec computed from non-`None` input arg (`rate` or `log_rate`)."""
    with self._name_and_control_scope(name or 'rate_parameter'):
      return self._rate_parameter_no_checks()

  def _rate_parameter_no_checks(self):
    if self._rate is None:
      return tf.exp(self._log_rate)
    return tensor_util.identity_as_tensor(self._rate)

  def log_rate_parameter(self, name=None):
    """Log-rate vec computed from non-`None` input arg (`rate`, `log_rate`)."""
    with self._name_and_control_scope(name or 'log_rate_parameter'):
      return self._log_rate_parameter_no_checks()

  def _log_rate_parameter_no_checks(self):
    if self._log_rate is None:
      return tf.math.log(self._rate)
    return tensor_util.identity_as_tensor(self._log_rate)

  def _default_event_space_bijector(self):
    return

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'log_rate': tf.math.log(tf.reduce_mean(value, axis=0))}

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if self._rate is not None:
      if is_init != tensor_util.is_ref(self._rate):
        assertions.append(assert_util.assert_non_negative(
            self._rate,
            message='Argument `rate` must be non-negative.'))
    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    return assertions
```

`tfp.distributions.Poisson` 继承自 `tfp.distribution.DiscreteDistributionMixin` 和 `tfp.distribution.AutoCompositeTensorDistribution`

# 四、对比分析
Pytorch 与 Tensorflow 实现方式大体类似，都是通过基本的概率计算得到相应的概率属性。Tensorflow 实现的方法更丰富，测试更为详细。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.poisson(rate)
```
参数 `rate` 为 Poisson 分布的参数。

例如，随机变量 $X$ 服从 Poisson 分布，即 $X \sim Poisson(\lambda)$ ，对应的参数 `rate`$=\lambda$。

## 底层OP设计
本次任务的设计思路与已有概率分布保持一致，不涉及底层 OP 的开发。

## API实现方案
新增 `Poisson` 类

```python
class Poisson(Distribution):
  def __init__(self, rate):
    super().__init__(batch_shape=self.rate.shape, event_shape=())
    
    ...
    
```

`Poisson` 类的初始化参数是 `rate` ，类包含的方法及实现方案如下：

记参数 `rate`$\lambda$。

- `mean` 计算均值

均值的计算方法： $\lambda$

- `variance` 计算方差

方差的计算方法： $\lambda$

- `entropy` 熵计算

熵的计算方法： $H = - \sum_x f(x) \log{f(x)}$

- `kl_divergence` 相对熵计算

相对熵的计算方法： $D_{KL}(\lambda_1, \lambda_2) = \sum_x f_1(x) \log{\frac{f_1(x)}{f_2(x)}}$

- `sample` 随机采样

采样方法： 需要自行设计poisson采样器

- `prob` 概率密度

概率密度计算方法： $f(x;\lambda) = \frac{e^{-\lambda} \cdot \lambda^x}{x!}$

- `log_prob` 对数概率密度

对数概率密度计算方法： $\log[f(x;\lambda)] = -\lambda + x \log \lambda - \log (x!)$



# 六、测试和验收的考量
`Poisson` 类测试以 Numpy 作为基准，验证API的正确性。
1. 使用 Numpy 实现所有 Poisson 的API，集成为 `PoissonNumpy` 类，用以验证本次任务开发的 API 的正确性。

2. 使用同样的参数实例化 `Poisson` 类和 `PoissonNumpy` 类，并调用 `mean`、`variance`、`entropy`、`prob`、`kl_divergence`方法，测试结果是否相等（容许一定误差）。参数 `rate` 的支持的数据类型需测试详尽。

3. 使用 `Poisson` 类的 `sample` 方法生成5000个样本，测试这些这样的均值和标准差是否正确。


# 七、可行性分析和排期规划
- 排期规划

10月13日~10月20日完成API开发与调试。

10月21日~10月28日完成测试代码的开发。

# 八、影响面
本次任务影响的模块如下：
1. `paddle.distribution` 

新增 poisson.py 文件。

2. `./test/distribution`

新增 test_distribution_poisson.py 和 test_distribution_poisson_static.py 文件。

# 名词解释
- Poisson 分布

若随机变量 $X \sim Poisson(\lambda)$，则 $X$ 的概率密度函数为
$$f(x;\lambda) = \frac{e^{-\lambda} \lambda^x}{x!}$$

# 附件及参考资料
1. [Tensorflow 的 Poisson 文档](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Poisson)

2. [Pytorch 的 Poisson 文档](https://pytorch.org/docs/stable/distributions.html#poisson)

3. [Numpy 的 Poisson 文档](https://numpy.org/doc/1.21/reference/random/generated/numpy.random.poisson.html)

# paddle.distribution.LogNormal 设计文档

|API名称 | LogNormal | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-09-07 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220907_api_design_for_log_normal.md<br> | 


# 一、概述
## 1、相关背景
在当前的 Paddle 框架中，`paddle.distribution` 目录内已经实现了一系列概率分布的 API，为了扩展现有的概率分布方案，本次任务计划实现 Log Normal 分布的 API。
## 2、功能目标
新增 LogNormal API，用于 Log Normal 分布的概率统计与随机采样，包括如下方法：
- `mean` 计算均值
- `variance` 计算方差
- `sample` 随机采样
- `rsample` 重参数化采
- `prob` 概率密度
- `log_prob` 对数概率密度
- `entropy` 熵计算
- `kl_divergence` 相对熵计算

## 3、意义
实现 LogNormal API，将能丰富 Paddle 的概率分布方案，进一步完善 Paddle 框架。
# 二、飞桨现状
Paddle 框架内定义了 `Distribution` 抽象基类，通过继承 `Distribution`，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 LogNormal 概率分布，需要单独开发实现，实现思路与其他概率分布的相同。LogNormal API 的具体实现会使用到许多基础 API，如 `paddle.exp`、`paddle.log`。

# 三、业内方案调研
PyTorch的 `LogNormal` 类是通过继承 `TransformedDistribution` 类实现，将Normal作为基础分布，变换得到Log Normal分布。
```python
class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
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

Tensorflow的 `LogNormal` 类也是通过继承 `transformed_distribution.TransformedDistribution` 类实现，将Normal分布变换得到Log Normal分布。
```python
class LogNormal(transformed_distribution.TransformedDistribution):
  """The log-normal distribution."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='LogNormal'):
    """Construct a log-normal distribution.

    The LogNormal distribution models positive-valued random variables
    whose logarithm is normally distributed with mean `loc` and
    standard deviation `scale`. It is constructed as the exponential
    transformation of a Normal distribution.

    Args:
      loc: Floating-point `Tensor`; the means of the underlying
        Normal distribution(s).
      scale: Floating-point `Tensor`; the stddevs of the underlying
        Normal distribution(s).
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      super(LogNormal, self).__init__(
          distribution=normal.Normal(loc=loc, scale=scale),
          bijector=exp_bijector.Exp(),
          validate_args=validate_args,
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
    """Distribution parameter for the pre-transformed mean."""
    return self.distribution.loc

  @property
  def scale(self):
    """Distribution parameter for the pre-transformed standard deviation."""
    return self.distribution.scale

  experimental_is_sharded = False

  @classmethod
  def experimental_from_mean_variance(cls, mean, variance, **kwargs):
    """Constructs a LogNormal from its mean and variance.

    **Experimental: Naming, location of this API may change.**

    Args:
      mean: The mean of the constructed distribution. Must be greater than 0.
      variance: The variance of the distribution. Must be greater than 0.
      **kwargs: Other keyword arguments passed directly to `__init__`, e.g.
        `validate_args`.

    Returns:
      lognormal: A distribution with the given parameterization.
    """
    dtype = dtype_util.common_dtype([mean, variance], dtype_hint=tf.float32)
    mean = tensor_util.convert_nonref_to_tensor(mean, dtype=dtype)
    variance = tensor_util.convert_nonref_to_tensor(variance, dtype=dtype)

    scale = DeferredTensor(
        mean, lambda mean: tf.sqrt(tf.math.log1p(variance / mean ** 2)))
    loc = DeferredTensor(
        mean, lambda mean: tf.math.log(mean) - scale ** 2 / 2.)
    return cls(loc=loc, scale=scale, **kwargs)

  def _log_prob(self, x):
    answer = super(LogNormal, self)._log_prob(x)
    # The formula inherited from TransformedDistribution computes `nan` for `x
    # == 0`.  However, there's hope that it's not too inaccurate for small
    # finite `x`, because `x` only appears as `log(x)`, and `log` is effectively
    # discontinuous at 0.  Furthermore, the result should be dominated by the
    # `log(x)**2` term, with no higher-order term that needs to be cancelled
    # numerically.
    return tf.where(tf.equal(x, 0.0),
                    tf.constant(-np.inf, dtype=answer.dtype),
                    answer)

  def _mean(self):
    return tf.exp(self.distribution.mean() + 0.5 * self.distribution.variance())

  def _variance(self):
    variance = self.distribution.variance()
    return (tf.math.expm1(variance) *
            tf.exp(2. * self.distribution.mean() + variance))

  def _mode(self):
    return tf.exp(self.distribution.mean() - self.distribution.variance())

  def _entropy(self):
    return (self.distribution.mean() + 0.5 +
            tf.math.log(self.distribution.stddev()) + 0.5 * np.log(2 * np.pi))

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _default_event_space_bijector(self):
    return exp_bijector.Exp(validate_args=self.validate_args)

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    log_x = tf.math.log(value)
    return {'loc': tf.reduce_mean(log_x, axis=0),
            'scale': tf.math.reduce_std(log_x, axis=0)}


@kullback_leibler.RegisterKL(LogNormal, LogNormal)
def _kl_lognormal_lognormal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b LogNormal.

  This is the same as the KL divergence between the underlying Normal
  distributions.

  Args:
    a: instance of a LogNormal distribution object.
    b: instance of a LogNormal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_lognormal_lognormal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  return kullback_leibler.kl_divergence(
      a.distribution,
      b.distribution,
      name=(name or 'kl_lognormal_lognormal'))
```

# 四、对比分析
Pytroch 和 Tensorflow 实现 Log Normal分布的方式基本相同，都是将 Normal 分布进行变换，得到 LogNormal 分布。Tensorflow 实现的方法更丰富，测试更为详细。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.LogNormal(loc, scale)
```
参数 `loc`, `scale` 分别为基础分布 Normal 的均值和标准差。

例如，随机变量 $X$ 服从 Log Normal 分布，即 $lnX \sim N(\mu, \sigma^2)$ ，对应的参数 $loc=\mu$ ， $scale=\sigma$ 。

## 底层OP设计
本次任务的设计思路与已有概率分布保持一致，不涉及底层 OP 的开发。

## API实现方案

1. 新增 `LogNormal` 类

由于 `paddle.distribution` 中已有 Normal 分布，任务计划通过继承 `TransformedDistribution` 基类实现 `LogNormal` 类 ，将 Normal 分布进行变换得到 Log Normal 分布。
```python
class LogNormal(TransformedDistribution):
  def __init__(self, loc, scale):
    base_dist = Normal(loc, scale)
    super(LogNormal, self).__init__(base_dist, [paddle.distribution.ExpTransform()])
    
    ...
    
```

`LogNormal` 类的初始化参数有 $loc$ 和 $scale$ ，类包含的方法及实现方案如下：

记参数 $loc=\mu$ ， $scale=\sigma$ 。

- `mean` 计算均值

均值的计算方法： $e^ {\mu + \frac{\sigma^2}{2}}$

- `variance` 计算方差

方差的计算方法： $e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$

- `entropy` 熵计算

熵的计算方法：基础分布 Normal 的熵与 $\mu$ 求和。

- `kl_divergence` 相对熵计算

与对应的基础分布Normal的计算逻辑相同，复用即可。

- `sample` 随机采样

继承父类 `TransformedDistribution` 的 `sample` 方法，将基础分布 Normal 的 `sample` 结果进行变换，得到 Log Normal 分布的 `sample `结果。

以下的方法也是通过继承父类 `TransformedDistribution` 相应的方法实现。

- `rsample` 重参数化采样
- `prob` 概率密度
- `log_prob` 对数概率密度

2. 扩展 `Normal` 类

在 `Normal` 类中增加 `rsample` 方法，以支持 Normal 分布的重参数化采样。

3. 扩展 `TransformedDistribution` 类

在 `TransformedDistribution` 基类中增加 `rsample` 方法，以支持 `rsample` 的变换。

# 六、测试和验收的考量
`LogNormal` 类测试以 Numpy 作为基准，验证API的正确性。
1. 使用 Numpy 实现所有 Log Normal 的API，集成为 `LogNormalNumpy` 类，用以验证本次任务开发的 API 的正确性。

2. 使用同样的参数实例化 `LogNormal` 类和 `LogNormalNumpy` 类，并调用 `mean`、`variance`、`entropy`、`prob`、`kl_divergence`方法，测试结果是否相等（容许一定误差）。参数 `loc` 和 `scale` 的支持的数据类型需测试详尽。

3. 使用 `LogNormal` 类的 `sample` 方法生成6000个样本，测试这些这样的均值和标准差是否正确。

`Normal` 类已有 `NormalTest` 测试类，需增加 `rsample` 的测试。

# 七、可行性分析和排期规划
- 可行性分析

`paddle.distribution` 内定义了概率分布的基类，并且提供了较多的概率分布方案。参照已有方案的实现方法，可以增加新的 Log Normal 概率分布。

- 排期规划

9月1日~9月8日完成API开发与调试。

9月9日~9月16日完成测试代码的开发。

# 八、影响面
本次任务影响的模块如下：
1. `paddle.distribution` 

新增 log_normal.py 文件，修改 transformed_distribution.py 和 normal.py 文件。

2. `paddle.fluid.tests.unittests.distribution`

新增 test_distribution_log_normal.py 和 test_distribution_log_normal_static.py 文件。

# 名词解释
- Normal分布

若随机变量 $X$ 服从均值为 $\mu$ ，方差为 $\sigma^2$ 的 Normal 分布，则 $X$ 的概率密度函数为
$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^{2}}}$$

并记作 $X \sim N(\mu, \sigma^2)$ 。
- Log Normal分布

对于随机变量 $X$ ，若满足 $lnX\sim N(\mu, \sigma^2)$，则称随机变量 $X$ 服从 Log Normal 分布， $X$ 的概率密度函数为

$$f(x) = \frac{1}{\sigma x \sqrt{2\pi}}
                         e^{(-\frac{(ln(x)-\mu)^2}{2\sigma^2})}$$
                         
并记作 $lnX \sim N(\mu, \sigma^2)$ 。
# 附件及参考资料
1. [Tensorflow 的 LogNormal 文档](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogNormal)

2. [Pytorch 的 LogNormal 文档](https://pytorch.org/docs/stable/distributions.html#lognormal)

3. [Numpy 的 LogNormal 文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html)

4. [Tensorflow 的 LogNormal 测试代码](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/lognormal_test.py)

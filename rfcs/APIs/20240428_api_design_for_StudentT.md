# paddle.distribution.StudentT 设计文档

|API名称 | paddle.distribution.StudentT | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-04-28 | 
|版本号 | V1.1 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240428_api_design_for_StudentT.md<br> | 


# 一、概述
## 1、相关背景
在机器学习中, 概率编程是一个重要的分支, 它常被用于贝叶斯推理和一般的统计推断, 因此需要提升飞桨的概率分布 API 丰富度, 补充 `paddle.distribution.StudentT` 概率分布类的 API。

t分布是一种基础的概率分布, 它在概率论及统计学中用于根据小样本来估计总体呈正态分布且标准差未知的期望。学生t分布随机变量可以看作由相互独立的一个标准正态分布随机变量和卡方分布随机变量的比值构成；Location-scale t distribution随机变量由学生t分布随机变量经过线性变换得到。


## 2、功能目标
参考 Paddle 现有 distribution，增加 StudentT 分布类的概率统计与随机采样，包括如下方法：
- mean 计算均值
- variance 计算方差
- sample 随机采样
- prob 概率密度
- log_prob 对数概率密度
- entropy 熵计算

## 3、意义
丰富 Paddle 能够提供的分布类型，进一步完善 Paddle 框架以用于概率编程。

# 二、飞桨现状
Paddle 框架内定义了 Distribution 抽象基类，通过继承 Distribution，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 StudentT 概率分布，需要单独开发实现。

# 三、业内方案调研
### Pytorch
引用自：https://pytorch.org/docs/stable/distributions.html#studentt

`torch.distributions.studentT.StudentT(df, loc=0.0, scale=1.0, validate_args=None)`

```python
class StudentT(Distribution):
    r"""
    Creates a Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m = self.loc.clone(memory_format=torch.contiguous_format)
        m[self.df <= 1] = nan
        return m

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        m = self.df.clone(memory_format=torch.contiguous_format)
        m[self.df > 2] = (
            self.scale[self.df > 2].pow(2)
            * self.df[self.df > 2]
            / (self.df[self.df > 2] - 2)
        )
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self._chi2 = Chi2(self.df)
        batch_shape = self.df.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape=torch.Size()):
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(df)
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df)
        return self.loc + self.scale * Y


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z


    def entropy(self):
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )

```

`torch.distributions.studentT.StudentT`继承自 `torch.distributions.Distribution`

### TensorFlow
TensorFlow 中有 API `tfp.distributions.StudentT(
    df,
    loc,
    scale,
    validate_args=False,
    allow_nan_stats=True,
    name='StudentT'
)`

```python
class StudentT(distribution.AutoCompositeTensorDistribution):
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

    pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
    where,
    y = (x - mu) / sigma
    Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))

  """
  
  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='StudentT'):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
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
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
      super(StudentT, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
    return sample_n(
        n,
        df=df,
        loc=loc,
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, value):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    return log_prob(value, df, loc, scale)

  def _cdf(self, value):
    df = tf.convert_to_tensor(self.df)
    return cdf(value, df, self.loc, self.scale)

  def _survival_function(self, value):
    df = tf.convert_to_tensor(self.df)
    return cdf(-value, df, -self.loc, self.scale)

  def _quantile(self, value):
    df = tf.convert_to_tensor(self.df)
    return quantile(value, df, self.loc, self.scale)

  def _entropy(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, scale=scale)
    return entropy(df, scale, batch_shape, self.dtype)

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    mean = loc * tf.ones(self._batch_shape_tensor(loc=loc),
                         dtype=self.dtype)
    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='mean not defined for components of df <= 1'),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      """)
  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = tf.where(df > 2., df - 2., tf.ones_like(df))
    # Abs(scale) superfluous.
    var = (tf.ones(self._batch_shape_tensor(df=df, scale=scale),
                   dtype=self.dtype)
           * tf.square(scale) * df / denom)
    # When 1 < df <= 2, variance is infinite.
    result_where_defined = tf.where(
        df > 2.,
        var,
        dtype_util.as_numpy_dtype(self.dtype)(np.inf))

    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          result_where_defined,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='variance not defined for components of df <= 1'),
      ], result_where_defined)

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector (consider one that
    # transforms away the heavy tails).
    return identity_bijector.Identity(validate_args=self.validate_args)
```

`tfp.distributions.StudentT` 继承自 `tfp.distribution.AutoCompositeTensorDistribution` 

### Scipy
scipy.stats.t = <scipy.stats._continuous_distns.t_gen object>

```python
class t_gen(rv_continuous):
    r"""A Student's t continuous random variable.

    For the noncentral t distribution, see `nct`.

    %(before_notes)s

    See Also
    --------
    nct

    Notes
    -----
    The probability density function for `t` is:

    .. math::

        f(x, \nu) = \frac{\Gamma((\nu+1)/2)}
                        {\sqrt{\pi \nu} \Gamma(\nu/2)}
                    (1+x^2/\nu)^{-(\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter
    :math:`\nu` (denoted ``df`` in the implementation) satisfies
    :math:`\nu > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("df", False, (0, np.inf), (False, False))]

    def _rvs(self, df, size=None, random_state=None):
        return random_state.standard_t(df, size=size)

    def _pdf(self, x, df):
        return _lazywhere(
            df == np.inf, (x, df),
            f=lambda x, df: norm._pdf(x),
            f2=lambda x, df: (
                np.exp(self._logpdf(x, df))
            )
        )

    def _logpdf(self, x, df):

        def t_logpdf(x, df):
            return (np.log(sc.poch(0.5 * df, 0.5))
                    - 0.5 * (np.log(df) + np.log(np.pi))
                    - (df + 1)/2*np.log1p(x * x/df))

        def norm_logpdf(x, df):
            return norm._logpdf(x)

        return _lazywhere(df == np.inf, (x, df, ), f=norm_logpdf, f2=t_logpdf)

    def _cdf(self, x, df):
        return sc.stdtr(df, x)

    def _sf(self, x, df):
        return sc.stdtr(df, -x)

    def _ppf(self, q, df):
        return sc.stdtrit(df, q)

    def _isf(self, q, df):
        return -sc.stdtrit(df, q)

    def _stats(self, df):
        # infinite df -> normal distribution (0.0, 1.0, 0.0, 0.0)
        infinite_df = np.isposinf(df)

        mu = np.where(df > 1, 0.0, np.inf)

        condlist = ((df > 1) & (df <= 2),
                    (df > 2) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: df / (df-2.0),
                      lambda df: np.broadcast_to(1, df.shape))
        mu2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        g1 = np.where(df > 3, 0.0, np.nan)

        condlist = ((df > 2) & (df <= 4),
                    (df > 4) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: 6.0 / (df-4.0),
                      lambda df: np.broadcast_to(0, df.shape))
        g2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        return mu, mu2, g1, g2

    def _entropy(self, df):
        if df == np.inf:
            return norm._entropy()

        def regular(df):
            half = df/2
            half1 = (df + 1)/2
            return (half1*(sc.digamma(half1) - sc.digamma(half))
                    + np.log(np.sqrt(df)*sc.beta(half, 0.5)))

        def asymptotic(df):
            # Formula from Wolfram Alpha:
            # "asymptotic expansion (d+1)/2 * (digamma((d+1)/2) - digamma(d/2))
            #  + log(sqrt(d) * beta(d/2, 1/2))"
            h = (norm._entropy() + 1/df + (df**-2.)/4 - (df**-3.)/6
                 - (df**-4.)/8 + 3/10*(df**-5.) + (df**-6.)/4)
            return h

        h = _lazywhere(df >= 100, (df, ), f=asymptotic, f2=regular)
        return h
```

# 四、对比分析
Pytorch、Tensorflow_probability、Scipy 实现方式大体类似, 由于t分布是连续的概率分布，相应的概率属性都可以通过基本的概率计算得到。而在 Tensorflow_probability 和 Scipy 中除了实现了mean, variance, log_prob, entropy 等方法外还实现了 cdf, survival function, quantile 等方法。

survival function 本质上等于 1 - cdf，且 Tensorflow_probability 和 Scipy 的实现都是通过 1 - cdf, 所以可以省略。tfp 中的 quantile 方法与 Scipy 中的 ppf 方法等价, 相当于 icdf。

t 分布的 cdf 与 icdf 主要是在统计学的假设检验问题中有重要价值, 而在机器学习的概率模型领域下，使用最为广泛的主要还是分布的 pdf 与 logpdf。另一方面由于 t 分布的 cdf 和 icdf 的实现相对复杂, 对 pdf 的积分函数与其积分函数的逆函数的实现需要借助 [incomplete beta function](https://dlmf.nist.gov/8.17) 和 inverse incomplete beta function，这两个在 paddle 中尚未实现，所以考虑暂时先不实现 cdf 与 icdf。

此外, kl 散度的解析函数较难推导出, 以下过程参考[A Novel Kullback-Leilber Divergence Minimization-Based Adaptive Student's t-Filter](https://www.researchgate.net/publication/335580775_A_Novel_Kullback-Leilber_Divergence_Minimization-Based_Adaptive_Student's_t-Filter)

$$D_{KL}(\nu_1, \mu_1, \sigma1 ,\nu_2, \mu_2, \sigma_2) = \int_{x \in \Omega} f_1(x) \log{\frac{f_1(x)}{f_2(x)}}dx = \mathbb{E}_{f1(x)}[\log f_1(x) - \log f_2(x)]$$

$$
\begin{align*}
D_{KL} & =  \mathbb{E}\_{f1(x)} \[ \log \Gamma(\frac{\nu_1+1}{2}) - \log \Gamma(\frac{\nu_2+1}{2}) + \frac{1}{2}\log\frac{\nu_2}{\nu_1} + \log\frac{\sigma_2}{\sigma_1} - \log\Gamma(\frac{\nu_1}{2}) + \log\Gamma(\frac{\nu_2}{2}) \\
& - \frac{\nu_1+1}{2}\log[1+(\frac{x-\mu_1}{\sigma_1})^2 / \nu_1] +  \frac{\nu_2+1}{2}\log[1+(\frac{x-\mu_2}{\sigma_2})^2 / \nu_2]\] \\
& = \log \Gamma(\frac{\nu_1+1}{2}) - \log \Gamma(\frac{\nu_2+1}{2}) + \frac{1}{2}\log\frac{\nu_2}{\nu_1} + \log\frac{\sigma_2}{\sigma_1} - \log\Gamma(\frac{\nu_1}{2}) + \log\Gamma(\frac{\nu_2}{2}) \\
& - \frac{\nu_1+1}{2} \mathbb{E}\_{f1(x)}\[\log[1 +(\frac{x-\mu_1}{\sigma_1})^2 / \nu_1]\] \\
& + \frac{\nu_2+1}{2} \mathbb{E}\_{f1(x)}\[\log[1 +(\frac{x-\mu_2}{\sigma_2})^2 / \nu_2]\]
\end{align*}
$$

        from the derivation of entropy, we have

$$ \mathbb{E}_{f(x)}\[\log[1 +(\frac{x-\mu}{\sigma})^2 / \nu] \] = \psi(\frac{1+\nu}{2}) - \psi(\frac{\nu}{2})$$

        therefore

$$ \begin{aligned}
D_{KL} & = \log \Gamma(\frac{\nu_1+1}{2}) - \log \Gamma(\frac{\nu_2+1}{2}) + \frac{1}{2}\log\frac{\nu_2}{\nu_1} + \log\frac{\sigma_2}{\sigma_1} - \log\Gamma(\frac{\nu_1}{2}) + \log\Gamma(\frac{\nu_2}{2}) \\
& - \frac{\nu_1+1}{2} [\psi(\frac{1+\nu_1}{2}) - \psi(\frac{\nu_1}{2})] \\
& + \frac{\nu_2+1}{2} \mathbb{E}\_{f1(x)}\[\log[1 +(\frac{x-\mu_2}{\sigma_2})^2 / \nu_2]\]
\end{aligned} $$

只能推导到这一步, 因此建议暂不实现 kl 散度方法。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.studentT.StudentT(df, loc=0.0, scale=1.0)
```
参数 `df`, `loc`, `scale` 分别为 t 分布的三个参数, 表示自由度, 平移变换量, 和缩放变换量。

数学表示例如，随机变量 $X$ 服从 t 分布，即 $X \sim t(df, loc, scale)$。

## 底层OP设计
用现有API组合实现


## API实现方案
新增 `StudentT` 类

```python
class StudentT(Distribution):
  def __init__(self, df, loc=0.0, scale=1.0):
    super().__init__(batch_shape=self.df.shape, event_shape=())
    ...
    
```

`StudentT` 类的初始化参数有 `df`，`loc` 和 `scale` ，需满足 `df>0`, `scale>0`

类包含的方法及实现方案如下：

- `mean` 计算均值

    均值的计算方法：
    
    df>1 时, 为 $loc$ ;   
    df<=1 时, 为 nan

- `variance` 计算方差

    方差的计算方法： 
    
    df>2 时, 为 $scale^2 \frac{df}{df-2}$ ;  
    2>=df>1 时，为 inf ;  
    df<=1 时, 为 nan

- `entropy` 熵计算

    熵的计算方法： 
    $H = - \int_{x \in \Omega} f(x) \log{f(x)} dx$

    参考：[Shannon Entropy and Mutual Information for Multivariate SkewElliptical Distributions](https://marcgenton.github.io/2013.ACG.SJS.pdf) p46 s2.4 The multivariate Student’s t distribution
    记 $\nu = df$, $\mu = loc$, $\sigma=scale$, $\psi(\cdot)$ 是 digamma 函数

$$
H = \log(\frac{\Gamma(\nu/2)\Gamma(1/2) \sigma \sqrt{\nu}}{\Gamma[(1+\nu)/2]}) + \frac{(1+\nu)}{2} \cdot [\psi[(1+\nu)/2] - \psi(\nu/2)]
$$

- `sample` 随机采样

    采样方法：
    记 $\nu = df$, $\mu = loc$, $\sigma=scale$
    若 $X \sim t(\nu, \mu=0, \sigma=1)$, 则
    $X = \frac{Z}{\sqrt{T/\nu}}$, 其中 $Z \sim N(0, 1)$, $T \sim \chi^2(\nu)$, Z 与 T 相互独立。

    由于目前 paddle 无卡方分布，需要将其替换为 gamma 分布的特例，即 $T \sim \chi^2(\nu) = Gamma(\frac{\nu}{2}, \frac{1}{2})$
    
    对一般的 $X^{\prime} \sim t(\nu, \mu, \sigma)$
    只需对 X 做线性变换，即 $X^{\prime} = \mu + \sigma X$

- `prob` 概率密度

    概率密度函数的计算方法： 
    记 $\nu = df$, $\mu = loc$, $\sigma=scale$
    $X \sim t(\nu, \mu, \sigma)$, 则
    $f(x;\nu, \mu, \sigma) = \frac{\Gamma[(\nu+1)/2]}{\sigma\sqrt{\nu\pi}\Gamma(\nu/2)[1+(\frac{x-\mu}{\sigma})^2/\nu]^{(1+\nu)/2}}$

- `log_prob` 对数概率密度

    对数概率密度函数的计算方法： 
    记 $\nu = df$, $\mu = loc$, $\sigma=scale$
    $\log[f(\nu, \mu, \sigma)] = \log\Gamma[(\nu+1)/2] - \log \sigma - 0.5 \log \nu - 0.5 \log \pi - \log \Gamma(\nu/2) - 0.5 (\nu+1)\log[1+(\frac{x-\mu}{\sigma})^2/\nu] $



# 六、测试和验收的考量
`StudentT` 类测试以 Scipy 作为辅助，验证API的正确性。
1. `mean` 和 `variance` 直接验证即可。

2. `entropy`、`prob`、`log_prob` 分别用 `scipy.stats.t.entropy`、`scipy.stats.t.pdf`、`scipy.stats.t.logpdf` 进行验证。

3. 使用 `StudentT` 类的 `sample` 方法生成5000个样本，测试这些这样的均值和标准差是否正确。(参考的是目前 `geometric`、`gumbel`、`laplace`、`lognormal`、`multinomial`、`normal` 的测试方法)


# 七、可行性分析和排期规划
- 排期规划

4月29日~5月6日完成API开发与调试。

5月6日~5月13日完成测试代码的开发。

# 八、影响面
本次任务不影响其他模块

# 名词解释
- StudentT 分布

若随机变量 $X \sim t(\nu, \mu, \sigma)$，则 $X$ 的概率密度函数为
$$f(x;\nu, \mu, \sigma) = \frac{\Gamma[(\nu+1)/2]}{\sigma\sqrt{\nu\pi}\Gamma(\nu/2)[1+(\frac{x-\mu}{\sigma})^2/\nu]^{(1+\nu)/2}}$$

# 附件及参考资料
1. [Tensorflow 的 StudentT 文档](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/StudentT)

2. [Pytorch 的 StudentT 文档](https://pytorch.org/docs/stable/distributions.html#studentt)

3. [Scipy 的 StudentT 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)

4. [A Novel Kullback-Leilber Divergence Minimization-Based Adaptive Student's t-Filter](https://www.researchgate.net/publication/335580775_A_Novel_Kullback-Leilber_Divergence_Minimization-Based_Adaptive_Student's_t-Filter)
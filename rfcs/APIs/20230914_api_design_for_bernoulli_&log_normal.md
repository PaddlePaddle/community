# paddle.bernoulli_ / paddle.log_normal_ / paddle.log_normal API 设计文档

| API 名称     | log_normal / log_normal_/ bernoulli_   |
| ------------ | --------------------------------- |
| 提交作者     | PommesPeter                       |
| 提交时间     | 2023-09-14                        |
| 版本号       | V1.0                              |
| 依赖飞桨版本 | develop                           |
| 文件名       | 20230914_api_design_for_bernoulli_&log_normal.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持随机分布生成相关 API，Paddle 需要扩充 API `paddle.bernoulli_`, `paddle.log_normal`, `paddle.log_normal_`。

## 2、功能目标

指定均值 `mean` 和方差 `std`，生成其指定大小 `shape` 和数据类型 `dtype` 的对数正态分布，新增 paddle.log_normal /log_normal_ API；功能预期如下：

1. paddle.bernoulli_(x, p=0.5) 可以 inplace 的修改输入 x，填充伯努利分布的值
2. paddle.Tensor.bernoulli_(p=0.5) 作为 paddle.bernoulli_ 的 Tensor 类方法使用
3. paddle.log_normal_(x, mean=1.0, std=2.0) 可以 inplace 的修改输入 x，填充对数正态分布的值
4. paddle.Tensor.log_normal_(mean=1.0, std=2.0) 作为 paddle.log_normal_ 的 Tensor 类方法使用
5. paddle.log_normal(mean=1.0, std=2.0, shape=None, dtype=None) 作为非 inplace 的API，可以创建一个对数正态分布的Tensor

## 3、意义

为 Paddle 增加随机生成对数正态分布函数，丰富 `paddle` 中随机分布相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `log_normal` 和 `log_normal_` API，无法方便地生成对数正态分布，以及 inplace 的方式修改输入 `x`，填充对应的对数正态分布的值。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `Tensor.bernoulli_(p=0.5, *, generator=None) → Tensor` 的 API。
PyTorch 中有 `torch.distributions.log_normal.LogNormal(loc, scale, validate_args=None)` 的 API。
PyTorch 中有 `torch.Tensor.log_normal_` 的 API，详细参数为 `Tensor.log_normal_(mean=1, std=2, *, generator=None)`。

### bernoulli_

在实现方法上，PyTorch 通过 C++ **Kernel** 实现 `bernoulli_` API。

> Tensor.bernoulli_(p=0.5, *, generator=None) → Tensor

参数说明：
- p (float) - `p` should either be a scalar or tensor containing probabilities to be used for drawing the binary random number.
- generator (torch.Generator, optional) – a pseudorandom number generator for sampling

实现代码：

```c++
void bernoulli_scalar_kernel(const TensorBase &self, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  int64_t seed;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    seed = generator->random();
  }
  int64_t n = self.numel();
  bool contig = self.is_contiguous();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "bernoulli_scalar_cpu_", [&] {
    at::Tensor tmp_int_tensor;
    if (std::is_same<scalar_t, int>::value && contig) {
      tmp_int_tensor = self;
    } else {
      tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
    }

    scalar_t *self_ptr = self.data_ptr<scalar_t>();
    int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();

    auto sample = [&](int64_t begin, int64_t end) {
      int64_t len = end - begin;
      if (len > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, begin);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, len,
          sample_int_ptr + begin, p);
        vslDeleteStream(&stream);

        // vectorized copy if using buffer and contiguous, i.e., being non-int
        // type and contiguous
        if (!std::is_same<scalar_t, int>::value && contig) {
          scalar_t *self_seg = self_ptr + begin;
          int* tmp_seg = sample_int_ptr + begin;
          at::vec::convert<int, scalar_t>(tmp_seg, self_seg, len);
        }
      }
    };

    parallel_for(0, n, /* grain_size= */ 800, sample);

    // copy_ if using buffer and non contiguous
    if (!contig) {
      OptionalTensorRef(self)->copy_(tmp_int_tensor);
    }
  });
}
```

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/native/cpu/DistributionKernels.cpp#L49-L98)

### LogNormal

> Creates a log-normal distribution parameterized by loc and scale where:

```math
X ~ Normal(loc, scale)
Y = exp(X) ~ LogNormal(loc, scale)
```

参数表：

- loc (float or Tensor) – mean of log of distribution
- scale (float or Tensor) – standard deviation of log of the distribution

在实现方法上，LogNormal 是调用普通正态分布 `Normal` Python API 进行指数转换后得到对数正态分布。其中 `Normal` 是调用 `torch.normal` 生成正态分布，`torch.nomral` 底层 C++ Kernel 执行。

实现代码：

[代码位置](https://github.com/pytorch/pytorch/blob/HEAD/torch/distributions/log_normal.py#L4)

```python
class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

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
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc

```

Kernel 代码：

```c++
template<typename RNG>
void log_normal_kernel(TensorIteratorBase& iter, double mean_, double std_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    // define lambda for log_normal transformation
    auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::log_normal<accscalar_t>(transformation::normal<accscalar_t>(rand, mean, std)));
    };
    normal_and_transform<scalar_t, accscalar_t, curand4_engine_calls>(iter, gen, log_normal_func);
   });
}

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIteratorBase& iter, double mean, double std, c10::optional<Generator> gen) {
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};
```

对于形如 `torch.log_normal` 的形式，PyTorch 未提供相关的 API。

### log_normal_

> Tensor.log_normal_(mean=1, std=2, *, generator=None)

在实现方法上，PyTorch 提供了 `log_normal` 的原位操作，用来自对数正态分布的数字样本填充 tensor，该分布由给定平均值 $\mu$ 和标准差 $\sigma$ 参数化。请注意，平均值和标准差是基础正态分布的平均值和标准差，而不是返回对数正态分布的平均值和标准差：

$$
f(x) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}}
$$

利用 C++ 实现的 Kernel，添加对于 inplace 操作的支持。

[inplace 支持代码位置](https://github.com/pytorch/pytorch/blob/HEAD/aten/src/ATen/functorch/BatchRulesRandomness.cpp#L459)

C++ 代码支持：

```c++
Tensor& log_normal_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return at::native::templates::log_normal_impl_<LogNormalStub, Generator>(self, mean, std, std::move(gen));
}

template<template<typename> class log_normal_kernel, typename RNG>
at::Tensor& log_normal_impl_(at::Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = TensorIterator::borrowing_nullary_op(self);
  log_normal_kernel<RNG>()(iter, mean, std, gen);
  return self;
}
```

## Scipy

### bernoulli_

未提供 bernoulli_ inplace 操作。

### lognorm

`scipy.stats.lognorm = <scipy.stats._continuous_distns.lognorm_gen object>`
> A lognormal continuous random variable.

The probability density function for lognorm is:

$$
f(x, s) = \frac{1}{sx\sqrt{2\pi}}\exp(-\frac{\log^2(x)}{2s^2})
$$

上述概率密度以 “标准化” 形式定义。要移动和 / 或缩放分布，请使用 loc 和 scale 参数。具体来说， lognorm.pdf (x, s, loc,scale) 等同于 lognorm.pdf (y, s) /scale 且 y = (x - loc)/scale 。请注意，移动分布的位置并不会使它成为 “非中心” 分布；一些分布的非中心概括可以在单独的类中获得。

假设正态分布的随机变量 X 具有均值 mu 和标准差 sigma。那么 Y = exp (X) 服从对数正态分布，s = sigma，scale = exp (mu)。

实现代码：

```python
class lognorm_gen(rv_continuous):
    r"""A lognormal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lognorm` is:

    .. math::

        f(x, s) = \frac{1}{s x \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2(x)}{2s^2}\right)

    for :math:`x > 0`, :math:`s > 0`.

    `lognorm` takes ``s`` as a shape parameter for :math:`s`.

    %(after_notes)s

    Suppose a normally distributed random variable ``X`` has  mean ``mu`` and
    standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
    distributed with ``s = sigma`` and ``scale = exp(mu)``.

    %(example)

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo("s", False, (0, np.inf), (False, False))]

    def _rvs(self, s, size=None, random_state=None):
        return np.exp(s * random_state.standard_normal(size))

    def _pdf(self, x, s):
        # lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
        return np.exp(self._logpdf(x, s))

    def _logpdf(self, x, s):
        return _lognorm_logpdf(x, s)

    def _cdf(self, x, s):
        return _norm_cdf(np.log(x) / s)

    def _logcdf(self, x, s):
        return _norm_logcdf(np.log(x) / s)

    def _ppf(self, q, s):
        return np.exp(s * _norm_ppf(q))

    def _sf(self, x, s):
        return _norm_sf(np.log(x) / s)

    def _logsf(self, x, s):
        return _norm_logsf(np.log(x) / s)

    def _stats(self, s):
        p = np.exp(s*s)
        mu = np.sqrt(p)
        mu2 = p*(p-1)
        g1 = np.sqrt(p-1)*(2+p)
        g2 = np.polyval([1, 2, 3, 0, -6.0], p)
        return mu, mu2, g1, g2

    def _entropy(self, s):
        return 0.5 * (1 + np.log(2*np.pi) + 2 * np.log(s))

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="""\
        When `method='MLE'` and
        the location parameter is fixed by using the `floc` argument,
        this function uses explicit formulas for the maximum likelihood
        estimation of the log-normal shape and scale parameters, so the
        `optimizer`, `loc` and `scale` keyword arguments are ignored.
        If the location is free, a likelihood maximum is found by
        setting its partial derivative wrt to location to 0, and
        solving by substituting the analytical expressions of shape
        and scale (or provided parameters).
        See, e.g., equation 3.1 in
        A. Clifford Cohen & Betty Jones Whitten (1980)
        Estimation in the Three-Parameter Lognormal Distribution,
        Journal of the American Statistical Association, 75:370, 399-404
        https://doi.org/10.2307/2287466
        \n\n""")
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)

        parameters = _check_fit_input_parameters(self, data, args, kwds)
        data, fshape, floc, fscale = parameters
        data_min = np.min(data)

        def get_shape_scale(loc):
            # Calculate maximum likelihood scale and shape with analytical
            # formulas unless provided by the user
            if fshape is None or fscale is None:
                lndata = np.log(data - loc)
            scale = fscale or np.exp(lndata.mean())
            shape = fshape or np.sqrt(np.mean((lndata - np.log(scale))**2))
            return shape, scale

        def dL_dLoc(loc):
            # Derivative of (positive) LL w.r.t. loc
            shape, scale = get_shape_scale(loc)
            shifted = data - loc
            return np.sum((1 + np.log(shifted/scale)/shape**2)/shifted)

        def ll(loc):
            # (Positive) log-likelihood
            shape, scale = get_shape_scale(loc)
            return -self.nnlf((shape, loc, scale), data)

        if floc is None:
            # The location must be less than the minimum of the data.
            # Back off a bit to avoid numerical issues.
            spacing = np.spacing(data_min)
            rbrack = data_min - spacing

            # Find the right end of the bracket by successive doubling of the
            # distance to data_min. We're interested in a maximum LL, so the
            # slope dL_dLoc_rbrack should be negative at the right end.
            # optimization for later: share shape, scale
            dL_dLoc_rbrack = dL_dLoc(rbrack)
            ll_rbrack = ll(rbrack)
            delta = 2 * spacing  # 2 * (data_min - rbrack)
            while dL_dLoc_rbrack >= -1e-6:
                rbrack = data_min - delta
                dL_dLoc_rbrack = dL_dLoc(rbrack)
                delta *= 2

            if not np.isfinite(rbrack) or not np.isfinite(dL_dLoc_rbrack):
                # If we never find a negative slope, either we missed it or the
                # slope is always positive. It's usually the latter,
                # which means
                # loc = data_min - spacing
                # But sometimes when shape and/or scale are fixed there are
                # other issues, so be cautious.
                return super().fit(data, *args, **kwds)

            # Now find the left end of the bracket. Guess is `rbrack-1`
            # unless that is too small of a difference to resolve. Double
            # the size of the interval until the left end is found.
            lbrack = np.minimum(np.nextafter(rbrack, -np.inf), rbrack-1)
            dL_dLoc_lbrack = dL_dLoc(lbrack)
            delta = 2 * (rbrack - lbrack)
            while (np.isfinite(lbrack) and np.isfinite(dL_dLoc_lbrack)
                   and np.sign(dL_dLoc_lbrack) == np.sign(dL_dLoc_rbrack)):
                lbrack = rbrack - delta
                dL_dLoc_lbrack = dL_dLoc(lbrack)
                delta *= 2

            # I don't recall observing this, but just in case...
            if not np.isfinite(lbrack) or not np.isfinite(dL_dLoc_lbrack):
                return super().fit(data, *args, **kwds)

            # If we have a valid bracket, find the root
            res = root_scalar(dL_dLoc, bracket=(lbrack, rbrack))
            if not res.converged:
                return super().fit(data, *args, **kwds)

            # If the slope was positive near the minimum of the data,
            # the maximum LL could be there instead of at the root. Compare
            # the LL of the two points to decide.
            ll_root = ll(res.root)
            loc = res.root if ll_root > ll_rbrack else data_min-spacing

        else:
            if floc >= data_min:
                raise FitDataError("lognorm", lower=0., upper=np.inf)
            loc = floc

        shape, scale = get_shape_scale(loc)
        if not (self._argcheck(shape) and scale > 0):
            return super().fit(data, *args, **kwds)
        return shape, loc, scale


lognorm = lognorm_gen(a=0.0, name='lognorm')
```

### lognorm_

Scipy 未提供 lognorm_ 形式的 API。

## TensorFlow

### bernoulli_

未提供 bernoulli_ inplace 操作。

### lognormal

Tensorflow 在[Tensorflow/Probability](https://github.com/tensorflow/probability/blob/v0.21.0/tensorflow_probability/python/distributions/lognormal.py#L36-L164) 这个库中实现了 LogNormal，使用 Python API 结合相关已有算子组合实现。

代码如下：

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
```

### lognormal_

# 四、对比分析

## bernoulli_

- 共同点
  - Scipy 和 Tensorflow 均使用 Python API 组合实现 LogNormal 类。

- 不同点
  - PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。

## log_normal

- 共同点
  - Scipy 和 Tensorflow 均使用 Python API 组合实现 LogNormal 类。
  - 都能根据输入的 tensor 计算出填充对数正态分布之后的结果。
  - 都能根据均值和方差随机生成对数正态分布。
  - 都有提供对 Python 的调用接口。
  - 方差和均值均支持 tensor 的输入。
- 不同点
  - PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。
  - Scipy 和 Tensorflow 是使用现有的 C++ 算子对应的 Python API 组合实现。

## log_normal_

- 共同点
  - Scipy 和 Tensorflow 均使用 Python API 组合实现 LogNormal 类。
  - 都能根据输入的 tensor 计算出填充对数正态分布之后的结果。
  - 都能根据均值和方差随机生成对数正态分布。
  - 都有提供对 Python 的调用接口。
  - 方差和均值均支持 tensor 的输入。
- 不同点
  - PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。

# 五、设计思路与实现方案

## 命名与参数设计

添加 Python API:

`paddle.bernoulli_(x, p=0.5, name=None)`

```python
paddle.bernoulli_(
    x: Tensor,
    p: float,
    name: str=None
)
```

| 参数名 | 类型 | 描述 |
| -- | -- | -- |
| x | Tensor | 用户输入的 tensor |
| p | float | 伯努利分布的概率 |
| name | str | 操作对应的名字 |

`paddle.log_normal(x, mean=0.0, std=1.0, shape=None, dtype='float64', name=None)`

```python
paddle.log_normal(
    x: Tensor,
    mean: float | Tensor,
    std: float | Tensor,
    shape: List | Tuple,
    dtype: str | np.dtype
    name: str=None
)
```

| 参数名 | 类型 | 描述 |
| -- | -- | -- |
| x | Tensor | 用户输入的 tensor |
| mean | float \| Tensor | 对数正态分布的均值 |
| std | float \| Tensor | 对数正态分布的方差 |
| shape | Lists \| Tuple | 对数正态分布的 tensor 形状 |
| dtype | str \| np.dtyp | 对数正态分布的 tensor 类型 |
| name | str | 操作对应的名字 |

`paddle.log_normal_(x, mean=0.0, std=1.0, name=None)`

```python
paddle.log_normal_(
    x: Tensor,
    mean: float | Tensor,
    std: float | Tensor,
    name: str=None
)
```

| 参数名 | 类型 | 描述 |
| -- | -- | -- |
| x | Tensor | 用户输入的 tensor |
| mean | float \| Tensor | 对数正态分布的均值 |
| std | float \| Tensor | 对数正态分布的方差 |
| name | str | 操作对应的名字 |

## 底层OP设计

不涉及

## API实现方案

该 API 实现于 `python/paddle/tensor/random.py`。

### bernoulli_

考虑到 `Paddle` 本身已经实现 `paddle.uniform_`，本方案考虑使用 Python API 实现。

使用 `paddle.uniform_` 生成一个正态分布，随后将正态分布中小于 `p` 的部分，生成对应的 mask 矩阵，与原 tensor 进行索引。

### log_normal

考虑到 `Paddle` 本身已经实现 `paddle.gaussian` 和 `paddle.exp` 两个高性能算子，即本方案可用 Python API 组合实现。

根据推导可得，若 $X$ 是正态分布，则 $\exp(X)$ 为对数正态分布。对数的概率密度函数为：

$$f(x;μ,σ)=\frac{1}{x\sigma\sqrt{2\pi}}\exp(-\frac{(\ln x-\mu)^2}{2\sigma^2})$$

若 $Y=\ln X\sim N(\mu, \sigma^2)$，则有 $X=e^Y$，故，$F(x)=P\{X\le x\}=P\{e^Y\le x\}=P\{Y\le\ln x\}=f(\ln x)$

从而，$f(x)=F'(x)=\frac{1}{x}f(\ln x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp(-\frac{(\ln x-\mu)^2}{2\sigma^2})$

因此，只需要对所求得的正态分布求自然指数 $e^x$ 即可得到对数正态分布。

### log_normal_

考虑 inplace 操作，可以使用 `normal_` 和 `exp_` 组合实现 `log_normal_`。

# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 numpy 作为参考标准
- 对不同 dtype 的输入数据 `x` 进行计算精度检验 (float32, float64)
- 输入输出的容错性与错误提示信息
- 输出 dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景

# 七、可行性分析和排期规划

方案主要依赖现有原理实现。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响

# 名词解释

无

# 附件及参考资料

[torch.Tensor.log_normal_](https://pytorch.org/docs/stable/generated/torch.Tensor.log_normal_.html)

[torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/stable/distributions.html#lognormal)

[scipy.stats.lognorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)

[tfp.distributions.LogNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogNormal)

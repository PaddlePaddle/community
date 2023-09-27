# paddle.distribution.multivariate_normal 设计文档

|API名称 | paddle.distribution.multivariate_normal | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-27 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20230927_api_design_for_multivariate_normal.md<br> | 


# 一、概述
## 1、相关背景
提升飞桨 API 丰富度, 需要扩充 API `paddle.distribution.multivariate_normal`。

## 2、功能目标
参考 Paddle 现有 distribution，增加 MultivariateNormal 分布类的概率统计与随机采样，包括如下方法：
- mean 计算均值
- variance 计算方差
- sample 随机采样
- rsample 重参数化随机采样
- prob 概率密度
- log_prob 对数概率密度
- entropy 熵计算
- kl_divergence 相对熵计算

## 3、意义
丰富 Paddle 能够提供的分布类型，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了 Distribution 抽象基类，通过继承 Distribution，框架实现了 Uniform、Normal 等概率分布。目前 Paddle 中暂无 MultivariateNormal 概率分布，需要单独开发实现，实现思路与其他概率分布的相同。

# 三、业内方案调研
### Pytorch
PyTorch 中有 API `torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None)`
```python
class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
```

`torch.distributions.multivariate_normal.MultivariateNormal`继承自 `torch.distributions.Distribution`

### TensorFlow
TensorFlow 中有 API `tfp.distributions.MultivariateNormalTriL(
    loc=None,
    scale_tril=None,
    validate_args=False,
    allow_nan_stats=True,
    experimental_use_kahan_sum=False,
    name='MultivariateNormalTriL'
)`

```python
class MultivariateNormalTriL(
    mvn_linear_operator.MultivariateNormalLinearOperator):

  def __init__(self,
               loc=None,
               scale_tril=None,
               validate_args=False,
               allow_nan_stats=True,
               experimental_use_kahan_sum=False,
               name='MultivariateNormalTriL'):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

    ```none
    scale = scale_tril
    ```

    where `scale_tril` is lower-triangular `k x k` matrix with non-zero
    diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_tril: Floating-point, lower-triangular `Tensor` with non-zero
        diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
        `b >= 0` and `k` is the event size.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values as well as
        when computing the log-determinant of the scale matrix. Doing so
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if neither `loc` nor `scale_tril` are specified.
    """
    parameters = dict(locals())
    if loc is None and scale_tril is None:
      raise ValueError('Must specify one or both of `loc`, `scale_tril`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale_tril], tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
      scale_tril = tensor_util.convert_nonref_to_tensor(
          scale_tril, name='scale_tril', dtype=dtype)
      self._scale_tril = scale_tril
      if scale_tril is None:
        scale = tf.linalg.LinearOperatorIdentity(
            num_rows=ps.dimension_size(loc, -1),
            dtype=loc.dtype,
            is_self_adjoint=True,
            is_positive_definite=True,
            assert_proper_shapes=validate_args)
      else:
        # No need to validate that scale_tril is non-singular.
        # LinearOperatorLowerTriangular has an assert_non_singular
        # method that is called by the Bijector.
        linop_cls = (KahanLogDetLinOpTriL if experimental_use_kahan_sum else
                     tf.linalg.LinearOperatorLowerTriangular)
        scale = linop_cls(
            scale_tril,
            is_non_singular=True,
            is_self_adjoint=False,
            is_positive_definite=False)
      super(MultivariateNormalTriL, self).__init__(
          loc=loc,
          scale=scale,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          experimental_use_kahan_sum=experimental_use_kahan_sum,
          name=name)
      self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        scale_tril=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=lambda sample_shape: ps.concat(
                [sample_shape, sample_shape[-1:]], axis=0),
            default_constraining_bijector_fn=lambda: fill_scale_tril_bijector.
            FillScaleTriL(diag_shift=dtype_util.eps(dtype))))
    # pylint: enable=g-long-lambda

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'loc': tf.reduce_mean(value, axis=0),
            'scale_tril': tf.linalg.cholesky(
                sample_stats.covariance(value, sample_axis=0, event_axis=-1))}

  @property
  def scale_tril(self):
    return self._scale_tril
```

`tfp.distributions.MultivariateNormalTriL` 继承自 `tfp.distribution.mvn_linear_operator.MultivariateNormalLinearOperator`

# 四、对比分析
Pytorch 与 Tensorflow 实现方式大体类似，都是通过基本的概率计算得到相应的概率属性。

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.distribution.multivariate_normal(loc, sacle)
```
参数 `loc`，`sacle`为 MultivariateNormal 分布的参数。

例如，随机变量 $X$ 服从 MultivariateNormal 分布，即 $X \sim MVN(\mu, \Sigma)$ ，对应的参数 `loc`$=\mu$，`sacle`$=\Sigma$。

## 底层OP设计
本次任务的设计思路与已有概率分布保持一致，不涉及底层 OP 的开发。

## API实现方案
新增 `MultivariateNormal` 类

```python
class MultivariateNormal(Distribution):
  def __init__(self, loc, scale):
    super().__init__(batch_shape=self.loc.shape, event_shape=())
    
    ...
    
```

`MultivariateNormal` 类的初始化参数是 `loc`，`sacle` ，类包含的方法及实现方案如下：

记参数 `loc`$=\mu$，`sacle`$=\Sigma$。

- `mean` 计算均值向量

均值向量的计算方法： $ \mu $

- `variance` 计算协方差矩阵

协方差矩阵的计算方法： $ \Sigma $

- `entropy` 熵计算

熵的计算方法： $H = - \sum_x f(x) \log{f(x)}$

- `kl_divergence` 相对熵计算

相对熵的计算方法： $D_{KL}(\mu_1, \mu_2, \Sigma_1, \Sigma_2) = - \sum_x f_1(x) \log{\frac{f_1(x)}{f_2(x)}}$

- `sample` 随机采样

采样方法： 通过standard normal采样后做仿射变换

- `rsample` 随机采样

采样方法： 通过standard normal采样后做仿射变换

- `prob` 概率密度

概率密度计算方法： $$f(X ;\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp(-\frac{1}{2}(X - \mu)^{\intercal} \Sigma^{-1} (X - \mu))$$

- `log_prob` 对数概率密度

对数概率密度计算方法： 概率密度取对数


# 六、测试和验收的考量
`MultivariateNormal` 类测试以 Numpy 作为基准，验证API的正确性。
1. 使用 Numpy 实现所有 MultivariateNormal 的API，集成为 `MultivariateNormalNumpy` 类，用以验证本次任务开发的 API 的正确性。

2. 使用同样的参数实例化 `MultivariateNormal` 类和 `MultivariateNormalNumpy` 类，并调用 `mean`、`variance`、`entropy`、`log_prob`、`kl_divergence`等方法，测试结果是否相等（容许一定误差）。参数 `rate` 的支持的数据类型需测试详尽。

3. 使用 `MultivariateNormal` 类的 `sample` 方法生成5000个样本，测试这些这样的均值和标准差是否正确。


# 七、可行性分析和排期规划
- 排期规划

10月29日~11月6日完成API开发与调试。

11月7日~10月14日完成测试代码的开发。

# 八、影响面
本次任务影响的模块如下：
1. `paddle.distribution` 

新增 multivariate_normal.py 文件。

2. `./test/distribution`

新增 test_distribution_multivariate_normal.py 和 test_distribution_multivariate_normal_static.py 文件。

# 名词解释
- MultivariateNormal 分布

若随机变量 $X \sim MVN(\mu, \Sigma)$，则 $X$ 的概率密度函数为
$$f(X ;\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp(-\frac{1}{2}(X - \mu)^{\intercal} \Sigma^{-1} (X - \mu))$$

# 附件及参考资料
1. [Pytorch 的 MultivariateNormal 文档](https://pytorch.org/docs/stable/distributions.html#multivariatenormal)

2. [Tensorflow 的 MultivariateNormal 文档](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/MultivariateNormalTriL)

3. [Numpy 的 MultivariateNormal 文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html)

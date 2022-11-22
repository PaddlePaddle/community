| API 名称 | paddle.distribution.MultivariateNormal|
| --- |-----------------------------------|
| 提交作者 | 王勇森(dasenCoding)|
| 提交时间 | 2022-10-31 |
| 版本号 | V1.0.0                            |
| 依赖飞桨版本 | V2.3.0                            |
| 文件名 | 20221027_api_design_for_MultivariateNormal.md |

# 一、概述
## 1、相关背景
多元统计分析中很多重要的理论和方法都是直接或间接地建立在正态分布基础上，许多统计量的极限分布往往和正态分布有关。

正态分布有极其广泛的实际背景，生产与科学实验中很多随机变量的概率分布都可以近似地用正态分布来进行描述。多变量正态分布亦称为多变量高斯分布。它是单维正态分布向多维的推广。它同矩阵正态分布有紧密的联系。

目前 Paddle 框架中已经集成了 Normal 分布，但还没有推广 MultivariateNormal 分布。所以此任务的目标是在 Paddle 框架中，基于现有概率分布方案，在其基础上进行扩展，新增 MultivariateNormal API，API 的调用路径为： `paddle.distribution.MultivariateNormal`。
## 2、功能目标

为 paddle 框架增加 API  `paddle.distribution.MultivariateNormal`，MultivariateNormal 表示多元正态分布，用于多元正态分布的概率统计与随机采样。API中包括了如下方法：

- `mean`计算均值；
- `variance`计算方差 ；
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `cdf`累积分布函数；
- `entropy` 熵计算；

## 3、意义

为 Paddle 增加用于多元正态分布的概率统计与随机采样函数，丰富 `paddle.distribution` 下的 API，丰富 paddle 框架。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.MultivariateNormal`
- API `paddle.distribution.Multinomial`的代码开发风格可以作为`paddle.distribution.MultivariateNormal` 的主要参考。


# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None)`

主要参数变量：


- **loc (**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – mean of the distribution**

- **covariance_matrix (**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – positive-definite covariance matrix**

- **precision_matrix (**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – positive-definite precision matrix**

- **scale_tril (**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – lower-triangular factor of covariance, with positive-valued diagonal**

### 源代码

```python
import math

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L


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
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            batch_shape = torch.broadcast_shapes(precision_matrix.shape[:-2], loc.shape[:-1])
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

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
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new


    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return (torch.matmul(self._unbroadcasted_scale_tril,
                             self._unbroadcasted_scale_tril.mT)
                .expand(self._batch_shape + self._event_shape + self._event_shape))

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(
            self._batch_shape + self._event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det


    def entropy(self):
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

```

## TensorFlow

TensorFlow 中包含 API `tfp.distributions.MultivariateNormalTriL`、`tfp.distributions.MultivariateNormalFullCovariance`、`tfp.distributions.MultivariateNormalDiagPlusLowRank`等。

其中 `tfp.distributions.MultivariateNormalFullCovariance` 使用的是协方差矩阵，和 pytorch 中使用相同。这里我们介绍 `MultivariateNormalFullCovariance`

主要参数变量包括：
```python
tfp.distributions.MultivariateNormalFullCovariance(
    loc=None,
    covariance_matrix=None,
    validate_args=False,
    allow_nan_stats=True,
    name='MultivariateNormalFullCovariance'
)
```

### 源代码

```python
class MultivariateNormalFullCovariance(mvn_tril.MultivariateNormalTriL):
  @deprecation.deprecated(
      '2019-12-01',
      '`MultivariateNormalFullCovariance` is deprecated, use '
      '`MultivariateNormalTriL(loc=loc, '
      'scale_tril=tf.linalg.cholesky(covariance_matrix))` instead.',
      warn_once=True)
  def __init__(self,
               loc=None,
               covariance_matrix=None,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateNormalFullCovariance'):
    """Construct Multivariate Normal distribution on `R^k`.
    The `batch_shape` is the broadcast shape between `loc` and
    `covariance_matrix` arguments.
    The `event_shape` is given by last dimension of the matrix implied by
    `covariance_matrix`. The last dimension of `loc` (if provided) must
    broadcast with this.
    A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
    definite matrix.  In other words it is (real) symmetric with all eigenvalues
    strictly positive.
    Additional leading dimensions (if any) will index batches.
    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      covariance_matrix: Floating-point, symmetric positive definite `Tensor` of
        same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
        is ignored, so if `covariance_matrix` is not symmetric no error will be
        raised (unless `validate_args is True`).  `covariance_matrix` has shape
        `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
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
      ValueError: if neither `loc` nor `covariance_matrix` are specified.
    """
    parameters = dict(locals())

    # Convert the covariance_matrix up to a scale_tril and call MVNTriL.
    with tf.name_scope(name) as name:
      with tf.name_scope('init'):
        dtype = dtype_util.common_dtype([loc, covariance_matrix], tf.float32)
        loc = loc if loc is None else tf.convert_to_tensor(
            loc, name='loc', dtype=dtype)
        if covariance_matrix is None:
          scale_tril = None
        else:
          covariance_matrix = tf.convert_to_tensor(
              covariance_matrix, name='covariance_matrix', dtype=dtype)
          if validate_args:
            covariance_matrix = distribution_util.with_dependencies([
                assert_util.assert_near(
                    covariance_matrix,
                    tf.linalg.matrix_transpose(covariance_matrix),
                    message='Matrix was not symmetric')
            ], covariance_matrix)
          # No need to validate that covariance_matrix is non-singular.
          # LinearOperatorLowerTriangular has an assert_non_singular method that
          # is called by the Bijector.
          # However, cholesky() ignores the upper triangular part, so we do need
          # to separately assert symmetric.
          scale_tril = tf.linalg.cholesky(covariance_matrix)
        super(MultivariateNormalFullCovariance, self).__init__(
            loc=loc,
            scale_tril=scale_tril,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
    self._parameters = parameters

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'loc': tf.reduce_mean(value, axis=0),
            'covariance_matrix': tfp_stats.covariance(value,
                                                      sample_axis=0,
                                                      event_axis=-1)}

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        covariance_matrix=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=lambda sample_shape: ps.concat(
                [sample_shape, sample_shape[-1:]], axis=0),
            default_constraining_bijector_fn=(lambda: chain_bijector.Chain([
                cholesky_outer_product_bijector.CholeskyOuterProduct(),
                fill_scale_tril_bijector.FillScaleTriL(
                    diag_shift=dtype_util.eps(dtype))
            ]))))
    # pylint: enable=g-long-lambda
```

## Numpy

Numpy 中包含 API `numpy.random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)`。

主要参数包括：

- **mean _1-D array_like, of length N_**

Mean of the N-dimensional distribution.

- **cov _2-D array_like, of shape (N, N)_**

Covariance matrix of the distribution. It must be symmetric and positive-semidefinite for proper sampling.

- **size _int or tuple of ints, optional_**

Given a shape of, for example, (m,n,k), m*n*k samples are generated, and packed in an m-by-n-by-k arrangement. Because each sample is N-dimensional, the output shape is (m,n,k,N). If no shape is specified, a single (N-D) sample is returned.

- **check_valid _{ ‘warn’, ‘raise’, ‘ignore’ }, optional**

Behavior when the covariance matrix is not positive semidefinite.

- **tol _float, optional_**

Tolerance when checking the singular values in covariance matrix. cov is cast to double before the check.

### 源代码

```python
def multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1, *args, **kwargs): # real signature unknown; NOTE: unreliably restored from __doc__ 
    """
    multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
    
            Draw random samples from a multivariate normal distribution.
    
            The multivariate normal, multinormal or Gaussian distribution is a
            generalization of the one-dimensional normal distribution to higher
            dimensions.  Such a distribution is specified by its mean and
            covariance matrix.  These parameters are analogous to the mean
            (average or "center") and variance (standard deviation, or "width,"
            squared) of the one-dimensional normal distribution.
    
            .. note::
                New code should use the ``multivariate_normal`` method of a ``default_rng()``
                instance instead; please see the :ref:`random-quick-start`.
    
            Parameters
            ----------
            mean : 1-D array_like, of length N
                Mean of the N-dimensional distribution.
            cov : 2-D array_like, of shape (N, N)
                Covariance matrix of the distribution. It must be symmetric and
                positive-semidefinite for proper sampling.
            size : int or tuple of ints, optional
                Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
                generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
                each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
                If no shape is specified, a single (`N`-D) sample is returned.
            check_valid : { 'warn', 'raise', 'ignore' }, optional
                Behavior when the covariance matrix is not positive semidefinite.
            tol : float, optional
                Tolerance when checking the singular values in covariance matrix.
                cov is cast to double before the check.
    
            Returns
            -------
            out : ndarray
                The drawn samples, of shape *size*, if that was provided.  If not,
                the shape is ``(N,)``.
    
                In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
                value drawn from the distribution.
    
            See Also
            --------
            Generator.multivariate_normal: which should be used for new code.
    
            Notes
            -----
            The mean is a coordinate in N-dimensional space, which represents the
            location where samples are most likely to be generated.  This is
            analogous to the peak of the bell curve for the one-dimensional or
            univariate normal distribution.
    
            Covariance indicates the level to which two variables vary together.
            From the multivariate normal distribution, we draw N-dimensional
            samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
            element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
            The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
            "spread").
    
            Instead of specifying the full covariance matrix, popular
            approximations include:
    
              - Spherical covariance (`cov` is a multiple of the identity matrix)
              - Diagonal covariance (`cov` has non-negative elements, and only on
                the diagonal)
    
            This geometrical property can be seen in two dimensions by plotting
            generated data-points:
    
            >>> mean = [0, 0]
            >>> cov = [[1, 0], [0, 100]]  # diagonal covariance
    
            Diagonal covariance means that points are oriented along x or y-axis:
    
            >>> import matplotlib.pyplot as plt
            >>> x, y = np.random.multivariate_normal(mean, cov, 5000).T
            >>> plt.plot(x, y, 'x')
            >>> plt.axis('equal')
            >>> plt.show()
    
            Note that the covariance matrix must be positive semidefinite (a.k.a.
            nonnegative-definite). Otherwise, the behavior of this method is
            undefined and backwards compatibility is not guaranteed.
    
            References
            ----------
            .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
                   Processes," 3rd ed., New York: McGraw-Hill, 1991.
            .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
                   Classification," 2nd ed., New York: Wiley, 2001.
    
            Examples
            --------
            >>> mean = (1, 2)
            >>> cov = [[1, 0], [0, 1]]
            >>> x = np.random.multivariate_normal(mean, cov, (3, 3))
            >>> x.shape
            (3, 3, 2)
    
            The following is probably true, given that 0.6 is roughly twice the
            standard deviation:
    
            >>> list((x[0,0,:] - mean) < 0.6)
            [True, True] # random
    """
    pass
```


# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

## 共同点

1. 三者均实现了 MultivariateNormal 分布功能，同时数学原理保持一致，仅是各自实现的方式不同。
2. 三者进行 MultivariateNormal 初始化时，均可使用位置向量 loc 以及协方差矩阵 covariance_matrix 进行初始化，且要求协方差矩阵为正定矩阵。
3. pytorch和tensorflow均实现了包括均值（mean），方差（variance），熵（entropy），对数概率密度（log_prob），模式（mode），累积分布密度函数（cdf），以及取样（sample）等方法和属性。
4. 三者均实现了对输入参数的检查，虽然各自所实现的方式不同，但总体的功能是类似的。

## 不同点

1. 初始化参数种类不同：pytorch 还可以使用 scale_tril、precision_matrix等参数替代 covariance_matrix 进行初始化，而 tensorflow 中并不能通过传入不同参数的方式来对分布进行初始化，仅能通过更换
API 接口来进行不同矩阵的初始化，为此 tensorflow 中提供了诸如：`tfp.distributions.MultivariateNormalTriL`、`tfp.distributions.MultivariateNormalFullCovariance`、`tfp.distributions.MultivariateNormalDiagPlusLowRank`等不同的初始化 API。
2. tensorflow 中提供了更多的方法属性，包括但不限于下面：
```python
cross_entropy(
    other, name='cross_entropy'
)

log_cdf(
    value, name='log_cdf', **kwargs
)

stddev(
    name='stddev', **kwargs
)

kl_divergence(
    other, name='kl_divergence'
)
```

3. numpy 不同于 pytorch 与 tensorflow，其仅提供了初始化参数需要的条件要求，没有提供MultivariateNormal的众多属性方法，仅是完成在给定 mean 与 covariance 下的分布表示。
4. 三种方式得到的 MultivariateNormal 的表示不同。比如 numpy 使用 ndarray 进行表示，pytorch 使用 tensor进行表示。
5. pytorch 支持重参数采样，numpy 与 tensorflow 中仅支持sample，如：
```python
rsample(sample_shape=torch.Size([]))
```
6. numpy 中 MultivariateNormal 的 mean 与 pytorch / tensorflow 的均值不同，如：
```python
# numpy
mean=(1,2)
cov = [[1,0],[0,1]]
x = np.random.multivariate_normal(mean,cov,(3,3))
x.mean() # 结果为：1.2373516168707248

#pytorch
from torch.distributions.multivariate_normal import MultivariateNormal as MN
x = MN(torch.tensor([1,2]),torch.tensor(cov))
x.mean # 结果为：tensor([1, 2])
```
7. numpy 中的 MultivariateNormal 仅是一个方法，并不是一个单独的类，因此没有定义属性和其他和前面两者类似的方法。在使用时，只需要指定 MultivariateNormal 需要的参数即可得到对应的分布，同时也可以再借助 Matplotlib 来展示 MultivariateNormal 分布。

## 方案优点

**pytorch:**
1. 集成的接口比较轻盈，同时实现了 MultivarianceNormal 的核心功能。
2. 源码以及实现思路清晰易懂，api 的使用较为方便。

**tensorflow:**
1. 集成的功能众多，几乎囊括了关于 MultivarianceNormal 的各个方面。

**numpy:**
1. numpy 更多是将 Multivariance 当作一种工具，需要使用时可以直接使用，没有再为其封装繁琐的方法和属性。

## 方案缺点
1. tensorflow 因集成的功能众多，所以在使用起来比较麻烦，需要花费更多的时间去学习使用。
2. numpy 则因其缺乏 MultivariateNormal 的其他属性方法，比如 prob / cdf / variance 等，可能会在使用时有缺陷。
3. pytorch 虽然整体简洁轻便，集成了 tensorflow 与 numpy 的各自优势，但是还缺乏一定的额外方法，比如：kl_divergence / log_cdf 等。

# 五、设计思路与实现方案

## 命名与参数设计
直接使用 MultivariateNormal 分布的名称作为此 API 的名称，参数保持 MultivariateNormal 分布最原生的参数即：

- loc：分布的均值
- covariance_matrix：正定协方差矩阵

预期 paddle 调用 MultivariateNormal API 的形式为：
```python
#loc: mean of the distribution
#covariance_matrix：positive-definite covariance matrix

paddle.distribution.multivariate_normal.MultivariateNormal(loc,covariance_matrix)
```
## 底层OP设计

使用 paddle 中现存的 API 进行实现，不考虑再令设计底层 OP。

## API实现方案

该 API 在 `paddle.distribution.multivariate_normal` 中实现，部分功能继承父类`Distribution`。
在经过调研对比后，MultivariateNormal API 中设定两个参数：loc 和 covariance_matrix，其中 loc 为分布均值，covariance_matrix 为协方差矩阵。

除了 API 调用的基本参数外，`paddle.distribution.multivariate_normal.MultivariateNormal` 中实现的属性、方法主要如下：

### 属性：
- mean ：分布均值
> 注：MultivariateNormal 分布的均值为 loc

- variance：方差
```python
matrix_decompos = paddle.linalg.cholesky(self.covariance_matrix).pow(2).sum(-1)
variance = paddle.expand(matrix_decompos,[self._batch_shape + self._event_shape])
```

- stddev：标准差
```python
paddle.sqrt(self.variance)
```

### 方法
- sample(shape)：随机采样  
在方法内部直接调用本类中的 rsample  方法。(参考pytorch复用重参数化采样结果):
```python
def sample(self, shape):
    sample(shape)
```

- rsample(shape)：重参数化采样

```python
def _batch_mv(bmat, bvec):
    bvec_unsqueeze = paddle.unsqueeze(bvec,1)
    bvec = paddle.squeeze(bvec_unsqueeze)
    return paddle.matmul(bmat,bvec)

def rsample(self, shape):
    shape = self._extend_shape(shape)
    eps = paddle.standard_normal(shape, dtype=None, name=None)
    unbroadcasted_scale_tril = paddle.linalg.cholesky(self.covariance_matrix)
  
    return self.loc + _batch_mv(unbroadcasted_scale_tril,eps)
```


- prob(value)：概率密度函数

其中要求：covariance_matrix 为非奇异正定矩阵。
```python
def prob(self, value):
    x = paddle.pow(2 * math.pi,-value.shape.pop(1) / 2) * paddle.pow(paddle.linalg.det(self.covariance_matrix), -1/2)
    y = paddle.exp(-1/2 * paddle.t(value - self.loc) * paddle.inverse(self.covariance_matrix) * (value - self.loc))
    return x * y
```

- log_prob(value)：对数概率密度函数
```python
def log_prob(self, value):
    return paddle.log(self.prob(value))
```

- entropy(value)：熵

```python
def entropy(self, value):
    sigma = paddle.linalg.det(self.convariance_matrix)
    return 1 / 2 * paddle.log(paddle.pow(2 * math.pi * math.e, value.shpe.pop(1)) * sigma)
```

- kl_divergence 两个MultivariateNormal分布之间的kl散度(other--MultivariateNormal类的一个实例):

```python
def kl_divergence(self, other):
  sector_1 = paddle.t(self.loc - other.loc) * paddle.inverse(other.convariance_matrix) * (self.loc - other.loc)
  sector_2 = paddle.log(paddle.linalg.det(paddle.inverse(other.convariance_matrix) * self.convariance_matrix))
  sector_3 = paddle.trace(paddle.inverse(other.convariance_matrix) * self.convariance_matrix)
  n = self.loc.shape.pop(1)
  return 0.5 * (sector_1 - sector_2 + sector_3 - n)
```
在`paddle/distribution/kl.py` 中注册`_kl_multivariatenormal_multivariatenormal`函数，使用时可直接调用`kl_divergence`计算`MultivariateNormal`分布之间的kl散度。

# 六、测试和验收的考量

`test_distribution_multivariate_normal`继承`unittest.TestCase`类中的方法，参考NormalTest的示例，新增一个`MultivariateNormalNumpy`类来验证`MultivariateNormal` API的正确性。
- 使用相同的参数实例化 `MultivariateNormal` 类和 `MultivariateNormalNumpy` 类，分别调用 `mean`、`variance`、`stddev`、`prob`、`log_prob`、`entropy`方法。将输出的结果进行对比，允许有一定的误差。
- 使用sample方法对多个样本进行测试。

1. 测试MultivariateNormal分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestMultivariateNormal继承unittest.TestCase，分别实现方法setUp（初始化），test_mean（mean单测），test_variance（variance单测），test_stddev（stddev单测），test_entropy（entropy单测），test_sample（sample单测）。

  * 均值、方差、标准差通过Numpy计算相应值，对比MultivariateNormal类中相应property的返回值，若一致即正确；
  
  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合MultivariateNormal分布。验证策略如下：随机采样30000个multivariate_normal分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.qmc.MultivariateNormalQMC`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于multivariate_normal分布，若计算所得ks值小于0.1，则拒绝不一致假设，两者属于同一分布；

2. 测试MultivariateNormal分布的概率密度函数

- 测试方法：该部分主要测试分布各种概率密度函数。类TestMultivariateNormalPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测）。

> 参考：community\rfcs\APIs\20220712_api_design_for_Laplace.md

# 七、可行性分析和排期规划

具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.multivariate_normal.MultivariateNormal` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.multivariate_normal.MultivariateNormal` API，与飞桨2.0代码风格保持一致

# 名词解释

##多元正态分布：MulitivariateNormal
多变量正态分布亦称为多变量高斯分布。它是单维正态分布向多维的推广。它同矩阵正态分布有紧密的联系。

矩阵常态分配（matrix normal distribution） 是一种几率分布，属于常态分配的之一。

> 参考：https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%85%83%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83 （中文维基百科）

# 附件及参考资料

## PyTorch

[torch.distributions.multivariate_normal.MultivariateNormal](https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal)

## TensorFLow

[tfp.distributions.MultivariateNormalFullCovariance](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance)

## Paddle

[paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/api/paddle/distribution/Normal_cn.html#normal)

## Numpy

[numpy.random.multivariate_normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html?highlight=multivariate_normal#numpy.random.multivariate_normal)
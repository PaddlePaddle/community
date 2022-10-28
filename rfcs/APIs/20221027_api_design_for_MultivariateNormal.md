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
- API `paddle.distribution.Normal`的代码开发风格可以作为`paddle.distribution.MultivariateNormal` 的主要参考。


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

TensorFlow 中包含 API `tfp.distributions.MultivariateNormalTriL`。

主要参数变量包括：
```python
tfp.distributions.MultivariateNormalTriL(
    loc=None,
    scale_tril=None,
    validate_args=False,
    allow_nan_stats=True,
    experimental_use_kahan_sum=False,
    name='MultivariateNormalTriL'
)
```

### 源代码

```python
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import stats as tfp_stats
from tensorflow_probability.python.bijectors import fill_scale_tril as fill_scale_tril_bijector
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic

from tensorflow.python.ops.linalg import linear_operator  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'MultivariateNormalTriL',
]


@linear_operator.make_composite_tensor
class KahanLogDetLinOpTriL(tf.linalg.LinearOperatorLowerTriangular):
  """Override `LinearOperatorLowerTriangular` logdet to use Kahan summation."""

  def _log_abs_determinant(self):
    return generic.reduce_kahan_sum(
        tf.math.log(tf.math.abs(self._get_diag())), axis=[-1]).total

class MultivariateNormalTriL(mvn_linear_operator.MultivariateNormalLinearOperator):
      def __init__(self,
                   loc=None,
                   scale_tril=None,
                   validate_args=False,
                   allow_nan_stats=True,
                   experimental_use_kahan_sum=False,
                   name='MultivariateNormalTriL'):
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
                    num_rows=distribution_util.dimension_size(loc, -1),
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
                    tfp_stats.covariance(value, sample_axis=0, event_axis=-1))}
    
      @property
      def scale_tril(self):
        return self._scale_tril
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

# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
## API实现方案

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
时间和开发排期规划，主要milestone

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料
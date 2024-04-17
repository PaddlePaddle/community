# paddle.distribution chi2/LKJCholesky 设计文档

| API 名称     | paddle.distribution.chi2 / paddle.distribution.LKJCholesky                     |
| ------------ | -------------------------------- |
| 提交作者     | cmcamdy                     |
| 提交时间     | 2024-04-15                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | develop                          |
| 文件名       | 20240415_api_design_for_Chi2_LKJCholesky.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，需要为飞桨扩充 API `paddle.distribution.chi2` 和 `paddle.distribution.LKJCholesky`

本 API 属于飞桨开源个人贡献赛 API 开发任务[NO.5 为 Paddle 新增 Chi2 / LKJCholesky API](https://github.com/PaddlePaddle/Paddle/issues/62905)的任务。

## 2、功能目标

- 实现卡方分布。调用路径为:paddle.distribution.chi2，作为独立的函数调用。
- 实现相关矩阵的下三角 Choleskey 因子的 LJK 分布。调用路径为:paddle.distribution.LKJCholesky，作为独立的函数调用。

## 3、意义

为飞桨增加更多概率分布，提升飞桨 API 丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

- PyTorch 中的 torch.distributions.chi2 [API文档](https://pytorch.org/docs/stable/distributions.html#chi2)
- PyTorch 中的 torch.distributions.chi2 [API 具体实现](https://github.com/pytorch/pytorch/blob/9bb54c7f3c048fdf2e5294e9e49e3642f1de98d8/torch/distributions/chi2.py)

- PyTorch 中的 torch.distributions.LKJCholesky [API文档](https://pytorch.org/docs/stable/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)
- PyTorch 中的 torch.distributions.LKJCholesky [API 具体实现](https://github.com/pytorch/pytorch/blob/9bb54c7f3c048fdf2e5294e9e49e3642f1de98d8/torch/distributions/lkj_cholesky.py)
- Pytorch 中的 [对于分布的测试代码](https://github.com/pytorch/pytorch/blob/e3ac61587aa368c613ef01df1f328a396b64cd5d/test/distributions/test_distributions.py)

## numpyro
- numpyro的 [Chi2 API文档](https://num.pyro.ai/en/stable/distributions.html#numpyro.distributions.continuous.Chi2)
- numpyro的 [Chi2 具体实现](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py#L508)
- numpyro的 [LKJCholesky API文档](https://num.pyro.ai/en/stable/distributions.html#lkjcholesky)
- numpyro的 [LKJCholesky 具体实现](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py#L938)

## 实现
### Chi2
- torch-chi2
```python
class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """
    arg_constraints = {"df": constraints.positive}

    def __init__(self, df, validate_args=None):
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    @property
    def df(self):
        return self.concentration * 2
```

- numpyro-Chi2
```python
class Chi2(Gamma):
    arg_constraints = {"df": constraints.positive}
    reparametrized_params = ["df"]

    def __init__(self, df, *, validate_args=None):
        self.df = df
        super(Chi2, self).__init__(0.5 * df, 0.5, validate_args=validate_args)
```

### LKJCholesky
- torch-LKJCholesky
```python
class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    """
    arg_constraints = {"concentration": constraints.positive}
    support = constraints.corr_cholesky

    def __init__(self, dim, concentration=1.0, validate_args=None):
        if dim < 2:
            raise ValueError(
                f"Expected dim to be an integer greater than or equal to 2. Found dim={dim}."
            )
        self.dim = dim
        (self.concentration,) = broadcast_all(concentration)
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = torch.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
            device=self.concentration.device,
        )
        offset = torch.cat([offset.new_zeros((1,)), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        super().__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim
        new.concentration = self.concentration.expand(batch_shape)
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        super(LKJCholesky, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # This uses the Onion method, but there are a few differences from [1] Sec. 3.2:
        # - This vectorizes the for loop and also works for heterogeneous eta.
        # - Same algorithm generalizes to n=1.
        # - The procedure is simplified since we are sampling the cholesky factor of
        #   the correlation matrix instead of the correlation matrix itself. As such,
        #   we only need to generate `w`.
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = torch.randn(
            self._extended_shape(sample_shape), dtype=y.dtype, device=y.device
        ).tril(-1)
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = torch.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        eps = torch.finfo(w.dtype).tiny
        diag_elems = torch.clamp(1 - torch.sum(w**2, dim=-1), min=eps).sqrt()
        w += torch.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        # See: https://mc-stan.org/docs/2_25/functions-reference/cholesky-lkj-correlation-distribution.html
        # The probability of a correlation matrix is proportional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # Additionally, the Jacobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) = prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i
        if self._validate_args:
            self._validate_sample(value)
        diag_elems = value.diagonal(dim1=-1, dim2=-2)[..., 1:]
        order = torch.arange(2, self.dim + 1, device=self.concentration.device)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = torch.sum(order * diag_elems.log(), dim=-1)
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = torch.lgamma(alpha) * dm1
        numerator = torch.mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term
```

- numpyro-LKJCholesky
```python
class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factors of correlation matrices. The distribution is
    controlled by ``concentration`` parameter :math:`\eta` to make the probability of the
    correlation matrix :math:`M` generated from a Cholesky factor propotional to
    :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over Cholesky factors of correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large diagonal entries
    (hence large determinent). This is useful when we know a priori that the underlying
    variables are not correlated.

    When ``concentration < 1``, the distribution favors samples with small diagonal entries
    (hence small determinent). This is useful when we know a priori that some underlying
    variables are correlated.

    Sample code for using LKJCholesky in the context of multivariate normal sample::

        def model(y):  # y has dimension N x d
            d = y.shape[1]
            N = y.shape[0]
            # Vector of variances for each of the d variables
            theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
            # Lower cholesky factor of a correlation matrix
            concentration = jnp.ones(1)  # Implies a uniform distribution over correlation matrices
            L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, concentration))
            # Lower cholesky factor of the covariance matrix
            sigma = jnp.sqrt(theta)
            # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
            L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)

            # Vector of expectations
            mu = jnp.zeros(d)

            with numpyro.plate("observations", N):
                obs = numpyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y)
            return obs

    :param int dimension: dimension of the matrices
    :param ndarray concentration: concentration/shape parameter of the
        distribution (often referred to as eta)
    :param str sample_method: Either "cvine" or "onion". Both methods are proposed in [1] and
        offer the same distribution over correlation matrices. But they are different in how
        to generate samples. Defaults to "onion".

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe
    """

    arg_constraints = {"concentration": constraints.positive}
    reparametrized_params = ["concentration"]
    support = constraints.corr_cholesky
    pytree_data_fields = ("_beta", "concentration")
    pytree_aux_fields = ("dimension", "sample_method")

    def __init__(
        self, dimension, concentration=1.0, sample_method="onion", *, validate_args=None
    ):
        if dimension < 2:
            raise ValueError("Dimension must be greater than or equal to 2.")
        self.dimension = dimension
        self.concentration = concentration
        batch_shape = jnp.shape(concentration)
        event_shape = (dimension, dimension)

        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2)
        # (up to a linear transform: x -> 2x - 1)
        Dm1 = self.dimension - 1
        marginal_concentration = concentration + 0.5 * (self.dimension - 2)
        offset = 0.5 * jnp.arange(Dm1)
        if sample_method == "onion":
            # The following construction follows from the algorithm in Section 3.2 of [1]:
            # NB: in [1], the method for case k > 1 can also work for the case k = 1.
            beta_concentration0 = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset
            )
            beta_concentration1 = offset + 0.5
            self._beta = Beta(beta_concentration1, beta_concentration0)
        elif sample_method == "cvine":
            # The following construction follows from the algorithm in Section 2.4 of [1]:
            # offset_tril is [0, 1, 1, 2, 2, 2,...] / 2
            offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (Dm1, Dm1)))
            beta_concentration = (
                jnp.expand_dims(marginal_concentration, axis=-1) - offset_tril
            )
            self._beta = Beta(beta_concentration, beta_concentration)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        self.sample_method = sample_method

        super(LKJCholesky, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _cvine(self, key, size):
        # C-vine method first uses beta_dist to generate partial correlations,
        # then apply signed stick breaking to transform to cholesky factor.
        # Here is an attempt to prove that using signed stick breaking to
        # generate correlation matrices is the same as the C-vine method in [1]
        # for the entry r_32.
        #
        # With notations follow from [1], we define
        #   p: partial correlation matrix,
        #   c: cholesky factor,
        #   r: correlation matrix.
        # From recursive formula (2) in [1], we have
        #   r_32 = p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} + p_21 * p_31 =: I
        # On the other hand, signed stick breaking process gives:
        #   l_21 = p_21, l_31 = p_31, l_22 = sqrt(1 - p_21^2), l_32 = p_32 * sqrt(1 - p_31^2)
        #   r_32 = l_21 * l_31 + l_22 * l_32
        #        = p_21 * p_31 + p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} = I
        beta_sample = self._beta.sample(key, size)
        partial_correlation = 2 * beta_sample - 1  # scale to domain to (-1, 1)
        return signed_stick_breaking_tril(partial_correlation)

    def _onion(self, key, size):
        key_beta, key_normal = random.split(key)
        # Now we generate w term in Algorithm 3.2 of [1].
        beta_sample = self._beta.sample(key_beta, size)
        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        normal_sample = random.normal(
            key_normal,
            shape=size
            + self.batch_shape
            + (self.dimension * (self.dimension - 1) // 2,),
        )
        normal_sample = vec_to_tril_matrix(normal_sample, diagonal=0)
        u_hypershere = normal_sample / jnp.linalg.norm(
            normal_sample, axis=-1, keepdims=True
        )
        w = jnp.expand_dims(jnp.sqrt(beta_sample), axis=-1) * u_hypershere

        # put w into the off-diagonal triangular part
        cholesky = jnp.zeros(size + self.batch_shape + self.event_shape)
        cholesky = cholesky.at[..., 1:, :-1].set(w)
        # correct the diagonal
        # NB: beta_sample = sum(w ** 2) because norm 2 of u is 1.
        diag = jnp.ones(cholesky.shape[:-1]).at[..., 1:].set(jnp.sqrt(1 - beta_sample))
        return add_diag(cholesky, diag)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        if self.sample_method == "onion":
            return self._onion(key, sample_shape)
        else:
            return self._cvine(key, sample_shape)

    @validate_sample
    def log_prob(self, value):
        # Note about computing Jacobian of the transformation from Cholesky factor to
        # correlation matrix:
        #
        #   Assume C = L@Lt and L = (1 0 0; a \sqrt(1-a^2) 0; b c \sqrt(1-b^2-c^2)), we have
        #   Then off-diagonal lower triangular vector of L is transformed to the off-diagonal
        #   lower triangular vector of C by the transform:
        #       (a, b, c) -> (a, b, ab + c\sqrt(1-a^2))
        #   Hence, Jacobian = 1 * 1 * \sqrt(1 - a^2) = \sqrt(1 - a^2) = L22, where L22
        #       is the 2th diagonal element of L
        #   Generally, for a D dimensional matrix, we have:
        #       Jacobian = L22^(D-2) * L33^(D-3) * ... * Ldd^0
        #
        # From [1], we know that probability of a correlation matrix is propotional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # On the other hand, Jabobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) =: prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i,
        # i = 2..D (we omit the element i = 1 because L_11 = 1)

        # Compute `order` vector (note that we need to reindex i -> i-2):
        one_to_D = jnp.arange(1, self.dimension)
        order_offset = (3 - self.dimension) + one_to_D
        order = 2 * jnp.expand_dims(self.concentration, axis=-1) - order_offset

        # Compute unnormalized log_prob:
        value_diag = jnp.asarray(value)[..., one_to_D, one_to_D]
        unnormalized = jnp.sum(order * jnp.log(value_diag), axis=-1)

        # Compute normalization constant (on the first proof of page 1999 of [1])
        Dm1 = self.dimension - 1
        alpha = self.concentration + 0.5 * Dm1
        denominator = gammaln(alpha) * Dm1
        numerator = multigammaln(alpha - 0.5, Dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * Dm1 * jnp.log(jnp.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized - normalize_term
```
# 四、对比分析
- `paddle.distribution.chi2`
    - `Pytorch`和`numpyro` 的`chi2`是通过继承`Gamma`实现的，`chi2`分布实际上等价于`Gamma(alpha=0.5*df, beta=0.5)`，在Paddle中已经有了[Gamma](https://github.com/PaddlePaddle/Paddle/blob/fba5029777f79c289003a24dbf736fdb6465d92a/python/paddle/distribution/gamma.py#L24)可以用相似的思路构造一下即可。

----------------------------------------
- `paddle.distribution.LKJCholesky`
    - `Pytorch`和`numpyro` 的`LKJCholesky`实现逻辑基本一致，区别是numpyro多了`sample method`的选择：`_onion`和`_cvine`。 `paddle` 中考虑实现后者。

# 五、设计思路与实现方案

`paddle` 目前的算子已经支持`Gamma`,基于此实现`chi2`即可。

参考`Pytorch`和`numpyro`中的实现方案实现LKJCholesky。

`Pytorch`中实现的`sample_method`只有`onion`，而`numpyro`则是实现了`onion`和`cvine`，相关实现的`sample`方法参考的是：https://ms.mcmaster.ca/canty/seminars/Joe_vinecorr_print.pdf

下面是这两种`sample_method`所对应的构造方法.


## onion

1.$Y \sim \text{Beta}(\alpha, \beta)$

2.$U_{\text{normal}}$是一个下三角矩阵，
$$
U_{\text{normal}} = 
\begin{cases} 
    \mathcal{N}(0,1), & \text{if } i > j \\
    0, & \text{if } i \leq j \\
\end{cases}
$$
3. 将这个下三角矩阵的每一行归一化到单位超球面上，得到$U_{\text{hypersphere}}$，
其中
$$
U_{\text{hypersphere},i,j} = \frac{U_{\text{normal},i,j}}{||U_{\text{normal},i}||}
$$

4. 用零填充$U_{\text{hypersphere}}$的第一行。

5. 计算$W$矩阵，它是$U_{\text{hypersphere}}$和$\sqrt{Y}$的哈达玛积（即元素相乘）。$W = \sqrt{Y} \cdot U_{\text{hypersphere}}$

$$
O_{i,j} = 
\begin{cases} 
    \sqrt{Y} \cdot U_{\text{hypersphere},i,j} , & \text{if } i > j \\
    \sqrt{1 - \sum_{k < i} W_{i,k}^2}, & \text{if } i = j \\
    0, & \text{if } i < j
\end{cases}
$$

其中$i$和$j$是矩阵的行索引和列索引，$U_{i,*}$表示$U$矩阵的第$i$行。这个过程生成的$O$矩阵是一个随机正交矩阵，它的行向量是彼此正交的，并且都有单位长度。



## cvine
- 

1.部分相关系数的生成

- 对于每一对变量，我们首先需要生成它们之间的部分相关系数。这可以通过从Beta分布中采样获得：

$$
Y \sim \text{Beta}(\beta, \beta)
$$

- 然后，将Beta分布的采样结果转换到$[-1, 1]$区间以获取部分相关系数：

$$
r_{ij} = 2y_{ij} - 1
$$

2.构造下三角矩阵

- 将这些部分相关系数填充到一个下三角矩阵中，其中$i > j$的元素对应于变量$i$和变量$j$之间的部分相关系数：

$$
R = 
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
r_{21} & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
r_{n1} & r_{n2} & \cdots & 1
\end{bmatrix}
$$

3.计算累积乘积的平方根

- 对于矩阵$R$中的每个元素，计算其累积乘积的平方根，并进行必要的填充：

$$
z_{ij} = 
\begin{cases}
 1 &, \text{if } i = 0\ or\ j=0 \\
 \sqrt{\prod_{k=0}^{j}(1-r_{ik}^2)} &, \text{if i>0  and j > 0} \\
\end{cases}
$$

- 这里，$z_{ij}$表示在考虑到变量$i$和变量$j$之间的直接依赖关系时。

4.最终矩阵的构造

-$out_{ij} =  z_{ij} * r_{ij}$





## 命名与参数设计

添加 Python API:

```python
paddle.distribution.chi2(df) 
```
参数表：
- `df (float or paddle.Tensor)`: 分布的形状参数，即自由度（degrees of freedom）。这个参数决定了分布的形状。它可以是一个浮点数或者一个 paddle.Tensor 对象。
----------------------------------------

```python
paddle.distribution.LKJCholesky(dim, concentration=1.0, sample_method = 'onion')
```
参数表：
- `dim (int)`: 目标相关矩阵的维度。
- `concentration (float, optional)`: 集中参数，默认为1.0。这个参数控制了生成的相关矩阵的分布。concentration 越大，生成的矩阵越接近单位矩阵。
- `sample_method (str)`:不同采样策略，可选项有：`onion` 和 `cvine`

## 底层 OP 设计

不涉及底层 OP。

# 六、测试和验收的考量

`paddle.distribution.chi2`, `paddle.distribution.LKJCholesky`：
 - 正确性验证：可以与 `Pytorch` 的结果对齐；
   - 不同 `shape；`
   - 前向计算；
 - 错误检查：输入数据类型不支持。

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关 PythonAPI 均有实现，可以在开源贡献个人挑战赛期间完成。
2024/04/15 - 2024/04/20 完成 API 主体实现；

2024/04/20 - 2024/04/25 完成单测；


# 八、影响面

对其他模块暂无影响

# 九、原型
- [chi2基本原型](https://github.com/cmcamdy/Develop_Diary/blob/master/Hackathon_doc/chi2/chi2.py)
- [lkj_cholesky基本原型](https://github.com/cmcamdy/Develop_Diary/blob/master/Hackathon_doc/lkj_cholesky/lkj_cholesky.py)
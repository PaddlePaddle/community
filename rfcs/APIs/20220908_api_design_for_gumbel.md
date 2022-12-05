| API 名称 | paddle.distribution.Gumbel |
| --- | --- |
| 提交作者 | 韩凡宇(PureNatural) 王勇森(dasenCoding) 周志峰 |
| 提交时间 | 2022-09-12 |
| 版本号 | V1.0.0 |
| 依赖飞桨版本 | V2.3.0 |
| 文件名 | 20220908_api_design_for_gumbel.md |

# 一、概述

## 1、相关背景
当前，GAN广泛应用于计算机视觉领域，相比而言，在NLP领域的应用还是相对较少，从离散分布中采样的数据是不可导的，在使用梯度下降算法时，无法正常更新模型中的参数，Gumbel 分布成功地使 GAN 摆脱了这个困境，为 GAN 在 NLP 领域地发展奠定了基础。
> 参考的原文链接：[https://blog.csdn.net/weixin_45753454/article/details/123694938](https://blog.csdn.net/weixin_45753454/article/details/123694938)


此任务的目标是在 Paddle 框架中，基于现有概率分布方案，在其基础上进行扩展，新增 Gumbel API，API 的调用路径为： `paddle.distribution.Gumbel`。

## 2、功能目标

为 paddle 框架增加 API  `paddle.distribution.Gumbel`，Gumbel 表示耿贝尔分布，用于耿贝尔分布的概率统计与随机采样。API中包括了如下方法：

- `mean`计算均值；
- `variance`计算方差 ；
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `cdf`累积分布函数；
- `entropy` 熵计算；

## 3、意义

为 Paddle 增加用于耿贝尔分布的概率统计与随机采样函数，丰富 `paddle.distribution` 下的 API，丰富 paddle 框架。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.Gumbel`
- API `paddle.distribution.Normal`的代码开发风格可以作为`paddle.distribution.Gumbel` 的主要参考。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `**torch.distributions.gumbel.Gumbel(**_**loc**_**, **_**scale**_**, **_**validate_args=None**_**`

主要参数变量：

- **loc (**[float](https://docs.python.org/3/library/functions.html#float)**_ __or__ _**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – Location parameter of the distribution**
- **scale (**[float](https://docs.python.org/3/library/functions.html#float)**_ or _**[Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)**) – Scale parameter of the distribution**

### 源代码

```python
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all, euler_constant


[docs]class Gumbel(TransformedDistribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, Number) and isinstance(scale, Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps)
        else:
            base_dist = Uniform(torch.full_like(self.loc, finfo.tiny),
                                torch.full_like(self.loc, 1 - finfo.eps))
        transforms = [ExpTransform().inv, AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
                      ExpTransform().inv, AffineTransform(loc=loc, scale=-self.scale)]
        super(Gumbel, self).__init__(base_dist, transforms, validate_args=validate_args)

[docs]    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super(Gumbel, self).expand(batch_shape, _instance=new)


    # Explicitly defining the log probability function for Gumbel due to precision issues
[docs]    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()


    @property
    def mean(self):
        return self.loc + self.scale * euler_constant

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

[docs]    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
```

## TensorFlow

TensorFlow 中包含 API `tfp.distributions.Gumbel`。

主要参数变量包括：
```python
tfp.distributions.Gumbel(
    loc,
    scale,
    validate_args=False,
    allow_nan_stats=True,
    name='Gumbel'
)
```



### 源代码



```python
class Gumbel(transformed_distribution.TransformedDistribution):
  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Gumbel'):
    """Construct Gumbel distributions with location and scale `loc` and `scale`.
    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).
    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s).
        scale must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'Gumbel'`.
    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype([loc, scale])
      # Positive scale is asserted by the incorporated Gumbel bijector.
      self._gumbel_bijector = gumbel_cdf_bijector.GumbelCDF(
          loc=loc, scale=scale, validate_args=validate_args)

      # Because the uniform sampler generates samples in `[0, 1)` this would
      # cause samples to lie in `(inf, -inf]` instead of `(inf, -inf)`. To fix
      # this, we use `np.finfo(dtype_util.as_numpy_dtype(self.dtype).tiny`
      # because it is the smallest, positive, 'normal' number.
      super(Gumbel, self).__init__(
          distribution=uniform.Uniform(
              low=np.finfo(dtype_util.as_numpy_dtype(dtype)).tiny,
              high=tf.ones([], dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          # The Gumbel bijector encodes the CDF function as the forward,
          # and hence needs to be inverted.
          bijector=invert_bijector.Invert(
              self._gumbel_bijector, validate_args=validate_args),
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
    """Distribution parameter for the location."""
    return self._gumbel_bijector.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._gumbel_bijector.scale

  experimental_is_sharded = False

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return 1. + tf.math.log(scale) + np.euler_gamma

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    z = (x - self.loc) / scale
    return -(z + tf.exp(-z)) - tf.math.log(scale)

  def _mean(self):
    return self.loc + self.scale * np.euler_gamma

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc) * np.pi / np.sqrt(6)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector. Consider switching to
    # Chain([Softplus(), Log()]) to lighten the doubly-exponential right tail.
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return self._gumbel_bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access
```



## Numpy

Numpy 中包含 API `random.gumbel(loc=0.0, scale=1.0, size=None)`。

主要参数包括：

- **loc：_float or array_like of floats, optional_**

The location of the mode of the distribution. Default is 0.

- **scale：_float or array_like of floats, optional_**

The scale parameter of the distribution. Default is 1. Must be non- negative.

- **size：_int or tuple of ints, optional_**

Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. If size is None 		(default), a single value is returned if loc and scale are both scalars. Otherwise, np.broadcast(loc, 			scale).size samples are drawn.
> 参考：[random.Generator.gumbel](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gumbel.html?highlight=gumbel#numpy.random.Generator.gumbel)

### 源代码

```python
def gumbel(loc=0.0, scale=1.0, size=None): # real signature unknown; restored from __doc__
    """
    gumbel(loc=0.0, scale=1.0, size=None)
    
            Draw samples from a Gumbel distribution.
    
            Draw samples from a Gumbel distribution with specified location and
            scale.  For more information on the Gumbel distribution, see
            Notes and References below.
    
            .. note::
                New code should use the ``gumbel`` method of a ``default_rng()``
                instance instead; please see the :ref:`random-quick-start`.
    
            Parameters
            ----------
            loc : float or array_like of floats, optional
                The location of the mode of the distribution. Default is 0.
            scale : float or array_like of floats, optional
                The scale parameter of the distribution. Default is 1. Must be non-
                negative.
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.  If size is ``None`` (default),
                a single value is returned if ``loc`` and ``scale`` are both scalars.
                Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    
            Returns
            -------
            out : ndarray or scalar
                Drawn samples from the parameterized Gumbel distribution.
    
            See Also
            --------
            scipy.stats.gumbel_l
            scipy.stats.gumbel_r
            scipy.stats.genextreme
            weibull
            Generator.gumbel: which should be used for new code.
    
            Notes
            -----
            The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
            Value Type I) distribution is one of a class of Generalized Extreme
            Value (GEV) distributions used in modeling extreme value problems.
            The Gumbel is a special case of the Extreme Value Type I distribution
            for maximums from distributions with "exponential-like" tails.
    
            The probability density for the Gumbel distribution is
    
            .. math:: p(x) = \frac{e^{-(x - \mu)/ \beta}}{\beta} e^{ -e^{-(x - \mu)/
                      \beta}},
    
            where :math:`\mu` is the mode, a location parameter, and
            :math:`\beta` is the scale parameter.
    
            The Gumbel (named for German mathematician Emil Julius Gumbel) was used
            very early in the hydrology literature, for modeling the occurrence of
            flood events. It is also used for modeling maximum wind speed and
            rainfall rates.  It is a "fat-tailed" distribution - the probability of
            an event in the tail of the distribution is larger than if one used a
            Gaussian, hence the surprisingly frequent occurrence of 100-year
            floods. Floods were initially modeled as a Gaussian process, which
            underestimated the frequency of extreme events.
    
            It is one of a class of extreme value distributions, the Generalized
            Extreme Value (GEV) distributions, which also includes the Weibull and
            Frechet.
    
            The function has a mean of :math:`\mu + 0.57721\beta` and a variance
            of :math:`\frac{\pi^2}{6}\beta^2`.
    
            References
            ----------
            .. [1] Gumbel, E. J., "Statistics of Extremes,"
                   New York: Columbia University Press, 1958.
            .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
                   Values from Insurance, Finance, Hydrology and Other Fields,"
                   Basel: Birkhauser Verlag, 2001.
    
            Examples
            --------
            Draw samples from the distribution:
    
            >>> mu, beta = 0, 0.1 # location and scale
            >>> s = np.random.gumbel(mu, beta, 1000)
    
            Display the histogram of the samples, along with
            the probability density function:
    
            >>> import matplotlib.pyplot as plt
            >>> count, bins, ignored = plt.hist(s, 30, density=True)
            >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
            ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
            ...          linewidth=2, color='r')
            >>> plt.show()
    
            Show how an extreme value distribution can arise from a Gaussian process
            and compare to a Gaussian:
    
            >>> means = []
            >>> maxima = []
            >>> for i in range(0,1000) :
            ...    a = np.random.normal(mu, beta, 1000)
            ...    means.append(a.mean())
            ...    maxima.append(a.max())
            >>> count, bins, ignored = plt.hist(maxima, 30, density=True)
            >>> beta = np.std(maxima) * np.sqrt(6) / np.pi
            >>> mu = np.mean(maxima) - 0.57721*beta
            >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
            ...          * np.exp(-np.exp(-(bins - mu)/beta)),
            ...          linewidth=2, color='r')
            >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))
            ...          * np.exp(-(bins - mu)**2 / (2 * beta**2)),
            ...          linewidth=2, color='g')
            >>> plt.show()
    """
    pass
```
# 四、对比分析
## 共同点

1. Pytorch 和 TensorFlow 都包含了Gumbel分布的基本参数 loc 和 scale，同时都在定义中添加了validate_args 参数，用来检查真实分布时参数的有效性；
1. Pytorch 和 TensorFlow 均实现了Gumbel 分布 entropy（熵）、mean（均值）、stddev（标准偏差）、variance（方差）以及计算 value 在给定的 Gumbel 分布中对应的概率的对数：log_prob() ；
1. 三者均实现了 Gumbel 分布的基本功能，同时所使用的数学原理基本是相同的；

## 不同点

1. 参数个数不同：tensorflow 中包含额外一个参数**allow_nan_stats ，其用来**统计数据的批次成员是否定义；
2. 参数要求不同：pytorch 中对于 loc 的要求为 real，而在 tensorflow 中对 loc 的要求必须是浮点型张量，
3. tensorflow 中额外添加了计算 CDF 和 逆 CDF 的方法：
```python
//计算累积概率密度
cdf(
    value, name='cdf', **kwargs
)

//计算逆概率密度
quantile(
    value, name='quantile', **kwargs
)


```

4. tensorflow 在提供了计算熵的方法基础上，额外添加了计算 cross_entropy （交叉熵）的方法：
```python
cross_entropy(
    other, name='cross_entropy'
)
```

5. pytorch 中可以使用 expand 来拓展 Gumbel 分布的维度：
```python
def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super(Gumbel, self).expand(batch_shape, _instance=new)

n.expand(torch.tensor([10]))
#result: Gumbel(loc: torch.Size([10]), scale: torch.Size([10]))
```

6. tensorflow 设置了直接查看变量属性的方法：
```python
@classmethod
parameter_properties(
    dtype=tf.float32, num_classes=None
)


dist.parameter_properties()
#result:
{'loc': ParameterProperties(event_ndims=0, 
                            event_ndims_tensor=0, 
                            shape_fn=<function ParameterProperties.<lambda> at 0x7f3c679178b0>, 
                            default_constraining_bijector_fn=<function _default_constraining_bijector_fn at 0x7f3c67917310>, 
                            is_preferred=True, 
                            is_tensor=True, 
                            specifies_shape=False),
 'scale': ParameterProperties(event_ndims=0, 
                              event_ndims_tensor=0, 
                              shape_fn=<function ParameterProperties.<lambda> at 0x7f3c679178b0>, 
                              default_constraining_bijector_fn=<function Gumbel._parameter_properties.<locals>.<lambda> at 0x7f3c5418e700>, 
							  is_preferred=True, 
							  is_tensor=True,
							  specifies_shape=False)
}
```

7. 除上面列出来的不同点外，tensorflow 还定义了其他方法比如：batch_shape_tensor、event_shape_tensor、kl_divergence、log_survival_function等等。
8. numpy 中的 Gumbel 仅是一个方法，并不是一个单独的类，因此没有定义属性和其他和前面两者类似的方法。在使用时，只需要指定 Gumbel 需要的参数即可得到对应的分布，同时也可以再借助 Matplotlib 来展示 Gumbel 分布：
```python
mu, beta = 0, 0.1 # location and scale
s = np.random.gumbel(mu, beta, 1000)
```
![image.png](https://cdn.nlark.com/yuque/0/2022/png/27690189/1662705228261-b073eba4-3c21-4ecd-8603-327ad9a3cb9a.png#clientId=ud385b8c3-d515-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=251&id=ua3a3a294&margin=%5Bobject%20Object%5D&name=image.png&originHeight=376&originWidth=952&originalType=binary&ratio=1&rotation=0&showTitle=false&size=35059&status=done&style=none&taskId=ucc8007ee-f84b-4c6d-a90b-83843782aa4&title=&width=634.6666666666666)
> 注：案例和图片参考：[numpy.random.gumbel](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html#)


# 五、方案设计

## 命名与参数设计

直接使用 Gumbel 分布的名称作为此 API 的名称，参数保持 Gumbel 分布最原生的参数即：位置参数 loc，以及尺度参数 scale，预期 paddle 调用 Gumbel API 的形式为：
```python
#loc: Location parameter of the distribution
#scale：Scale parameter of the distribution

paddle.distribution.Gumbel(loc,scale)
```

## 底层 OP 设计

使用 paddle 中现存的 API 进行实现，不考虑再令设计底层 OP。

## API 实现方案

该 API 在 `paddle.distribution.Gumbel` 中实现，部分功能继承于`TransformedDistribution`。使用`TransformedDistribution`的`ExpTransform`和`AffineTransform`将`Uniform`分布转换为`Gumbel`分布，`Uniform`分布paddle已经实现。
在经过调研对比后，Gumbel API 中设定两个参数：loc 和 scale，其中 loc 为位置参数，scale 为尺度参数；除了 API 调用的基本参数外，paddle.distribution.Gumbel 中实现的方法主要如下：

- mean ：均值
> 注：Gumbel 分布的均值为负欧拉常数：-γ

```python
self.loc + self.scale * np.euler_gamma
```

- variance：方差
```python
self.scale.pow(2) * math.pi.pow(2) / 6
```

- stddev：标准差
```python
math.sqrt(self.variance)
```

- sample(shape)：随机采样  
继承父类 TransformedDistribution 的 sample 方法
```python
self.sample(shape)
```

- rsample()：重参数化采样  
继承父类 TransformedDistribution 的 rsample 方法
```python
self.rsample(shape)
```

- prob(value)：概率密度函数
```python
exp(-(value - self.loc) / self.scale - exp(-(value - self.loc) / self.scale)) / self.scale
```

- log_prob(value)：对数概率密度函数
```python
np.log(self.probs(value))
```

- cdf(value)：累积分布函数
```python
paddle.exp(-paddle.exp(-(value - self.loc) / self.scale))
```

- entropy(scale)：熵
```python
self.scale.log() + (1 + np.euler_gamma)
```



# 六、测试和验收的考量
`GumbelTest`继承`unittest.TestCase`类中的方法，参考NormalTest的示例，新增一个`GumbelNumpy`类来验证`Gumbel` API的正确性。
- 使用相同的参数实例化 `Gumbel` 类和 `GumbelNumpy` 类，分别调用 `mean`、`variance`、`stddev`、`prob`、`log_prob`、`entropy`方法。将输出的结果进行对比，允许有一定的误差。
- 使用sample方法对多个样本进行测试。
- 对生成的样本集通过使用`pyplot`绘制直方图进行初步的检验;再通过调用`scipy.stats.kstest`方法进行KS检验,判断P值是否大于0.05。

# 七、可行性分析及规划排期
- 可行性分析

在`paddle`中，`TransformedDistribution`和`Uniform`已实现；同时`paddle` 已实现部分概率分布，可以对其进行参考实现新的 Gumbel 分布；

- 排期规划

9.12-9.15:完成API的开发及测试。  
9.16-9.17:完成中英文API文档的撰写。


# 八、影响面

为 paddle 增加了一个 `paddle.distribution.Gumbel` API，对在风格上与飞桨2.0代码保持一致，丰富了 paddle 框架，在处理 Gumbel 分布问题时更加方便高效。
- 新增`gumbel.py`,`test_distribution_gumbel.py`文件
- 在`transformed_distribution.py`中添加rsample方法

# 名词解释

## Gumbel 分布
- Gumbel 分布以 Emil Julius Gumbel(1891 - 1966)的名字命名，基于他描述分布的原始论文。


- Gumbel 分布主要用于对各种分布的多个样本的最大值或者最小值分布进行建模。例如，如果过去十年各年份河流最高水位的分布，它有助于预测发生极端地震、洪水或者其他自然灾害的可能性。


- Gumbel 分布是广义极值分布的特例，同时也被称为 WeiBull 分布和双指数分布Gumbel 分布表示最大值分布的潜在适用性与极值理论有关，这表明如果基础样本数据的分布是正态或者指数类型，Gumbel分布就是有用的。


## Uniform 分布
- Uniform 分布称为均匀分配，或者均匀分布，一种简单的概率分布，其分布为离散型均匀分布和连续型均匀分布两种类型的机率分布。
- 在概率论和统计学中，均匀分布也称为矩阵分布，它是对称概率分布，在相同长度间隔的分布概率是等可能的。

# 附件及参考资料

## PyTorch
[
](https://pytorch.org/docs/stable/distributions.html#laplace)

[torch.distributions.gumbel.Gumbel](https://pytorch.org/docs/stable/distributions.html#gumbel)

## TensorFlow
[
](https://www.tensorflow.org/api_docs/python/tf/compat/v1/distributions/Laplace)

[tfp.distributions.Gumbel](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Gumbel?hl=en#covariance)


## Numpy


[numpy.random.gumbel](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html#)

## Paddle

[paddle.distribution.Normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Normal_cn.html#normal)

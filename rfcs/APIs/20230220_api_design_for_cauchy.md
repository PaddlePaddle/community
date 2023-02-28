| API 名称 | paddle.distribution.Cauchy        |
| --- |-----------------------------------|
| 提交作者 | 王勇森(dasenCoding)                  |
| 提交时间 | 2023-02-23                        |
| 版本号 | V1.0.0                            |
| 依赖飞桨版本 | V2.4.1                            |
| 文件名 | 20230220_api_design_for_cauchy.md |

# 一、概述
## 1、相关背景
`Cauchy distribution` 也叫柯西-洛伦兹分布，它是以`奥古斯丁·路易·柯西`与`亨德里克·洛伦兹`名字命名的连续概率分布。其在自然科学中有着非常广泛的应用。

- 在物理学中，物理学家也将之称为`洛伦兹分布`或者`Breit-Wigner`分布，在物理学中的重要性很大一部分归因于它是描述受迫共振的微分方程的解。

- 在光谱学中，它描述了被共振或者其他机制加宽的谱线形状。

- 在量子世界，粒子和粒子距离很远，因此要显著描述电子的位置分布，只能是柯西-洛伦兹分布，不能用高斯分布刻画，因为高斯分布尺度不够，信号太弱，噪声将把电子的电磁能量淹没，模型无效。

`Cauchy distribution`的一个重要的特性是其均值和方差都不存在，并且没有高阶矩阵，正是这样的特性，使柯西分布常用于数据分析和贝叶斯统计学中非对称参数分布的情况。

目前 Paddle 框架中没有实现 Cauchy 分布。所以此任务的目标是在 Paddle 框架中，基于现有概率分布方案，在其基础上进行扩展，新增 Cauchy API，API 的调用路径为： `paddle.distribution.Cauchy`。
## 2、功能目标

为 paddle 框架增加 API  `paddle.distribution.Cauchy`，Cauchy 表示柯西分布，用于柯西分布的概率统计与随机采样。API中包括了如下方法：

- `mean`计算均值；
- `variance`计算方差 ；
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `entropy`熵计算；
- `cdf`累计概率密度函数；
- `kl` 两个分布间的kl散度；

上述方法可能无法全部支持，需要设计中说明不支持原因，抛出NotImplementedError异常即可。


## 3、意义

为 Paddle 增加用于柯西分布的概率统计与随机采样函数，丰富 `paddle.distribution` 下的 API，丰富 paddle 框架。

# 二、飞桨现状

- 目前 飞桨没有 API `paddle.distribution.Cauchy`
- API `paddle.distribution.Beta`的代码开发风格可以作为`paddle.distribution.Cauchy` 的主要参考。


# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.cauchy.Cauchy(loc, scale, validate_args=None)`

主要参数变量：


- **loc (**(float or Tensor)**) – mode or median of the distribution.**

- **scale  (** (float or Tensor) **) – half width at half maximum.**

### 源代码

```python
import math
from torch._six import inf, nan
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

class Cauchy(Distribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Cauchy, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    @property
    def mean(self):
        return torch.full(self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device)

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.full(self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -math.log(math.pi) - self.scale.log() - (1 + ((value - self.loc) / self.scale)**2).log()


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc


    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
```

## TensorFlow

TensorFlow 中包含 API `tfp.distributions.Cauchy`。

主要参数变量包括：
```python
tfp.distributions.Cauchy(
    loc,
    scale,
    validate_args=False,
    allow_nan_stats=True,
    name='Cauchy'
)
```

### 源代码

```python
class Cauchy(distribution.AutoCompositeTensorDistribution):
    def __init__(self,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Cauchy'):
      """Construct Cauchy distributions.
      The parameters `loc` and `scale` must be shaped in a way that supports
      broadcasting (e.g. `loc + scale` is a valid operation).
      Args:
        loc: Floating point tensor; the modes of the distribution(s).
        scale: Floating point tensor; the half-widths of the distribution(s) at
          their half-maximums. Must contain only positive values.
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
        TypeError: if `loc` and `scale` have different `dtype`.
      """
      parameters = dict(locals())
      with tf.name_scope(name) as name:
        dtype = dtype_util.common_dtype([loc, scale], tf.float32)
        self._loc = tensor_util.convert_nonref_to_tensor(
            loc, name='loc', dtype=dtype)
        self._scale = tensor_util.convert_nonref_to_tensor(
            scale, name='scale', dtype=dtype)
        dtype_util.assert_same_float_dtype([self._loc, self._scale])
        super(Cauchy, self).__init__(
            dtype=self._scale.dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
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
      return self._loc
  
    @property
    def scale(self):
      """Distribution parameter for the scale."""
      return self._scale
  
    def _event_shape_tensor(self):
      return tf.constant([], dtype=tf.int32)
  
    def _event_shape(self):
      return tf.TensorShape([])
  
    def _sample_n(self, n, seed=None):
      loc = tf.convert_to_tensor(self.loc)
      scale = tf.convert_to_tensor(self.scale)
      batch_shape = self._batch_shape_tensor(loc=loc, scale=scale)
      shape = ps.concat([[n], batch_shape], 0)
      probs = samplers.uniform(
          shape=shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
      return self._quantile(probs, loc=loc, scale=scale)
  
    def _log_prob(self, x):
      npdt = dtype_util.as_numpy_dtype(self.dtype)
      scale = tf.convert_to_tensor(self.scale)
      log_unnormalized_prob = -tf.math.log1p(tf.square(self._z(x, scale=scale)))
      log_normalization = npdt(np.log(np.pi)) + tf.math.log(scale)
      return log_unnormalized_prob - log_normalization
  
    def _cdf(self, x):
      return tf.atan(self._z(x)) / np.pi + 0.5
  
    def _log_cdf(self, x):
      return tf.math.log1p(2 / np.pi * tf.atan(self._z(x))) - np.log(2)
  
    def _entropy(self):
      h = np.log(4 * np.pi) + tf.math.log(self.scale)
      return h * tf.ones_like(self.loc)
  
    def _quantile(self, p, loc=None, scale=None):
      loc = tf.convert_to_tensor(self.loc if loc is None else loc)
      scale = tf.convert_to_tensor(self.scale if scale is None else scale)
      return loc + scale * tf.tan(np.pi * (p - 0.5))
  
    def _mode(self):
      return self.loc * tf.ones_like(self.scale)
  
    def _z(self, x, loc=None, scale=None):
      """Standardize input `x`."""
      loc = tf.convert_to_tensor(self.loc if loc is None else loc)
      scale = tf.convert_to_tensor(self.scale if scale is None else scale)
      with tf.name_scope('standardize'):
        return (x - loc) / scale
  
    def _inv_z(self, z):
      """Reconstruct input `x` from a its normalized version."""
      with tf.name_scope('reconstruct'):
        return z * self.scale + self.loc
  
    def _mean(self):
      if self.allow_nan_stats:
        return tf.fill(self.batch_shape_tensor(),
                       dtype_util.as_numpy_dtype(self.dtype)(np.nan))
      else:
        raise ValueError('`mean` is undefined for Cauchy distribution.')
  
    def _stddev(self):
      if self.allow_nan_stats:
        return tf.fill(self.batch_shape_tensor(),
                       dtype_util.as_numpy_dtype(self.dtype)(np.nan))
      else:
        raise ValueError('`stddev` is undefined for Cauchy distribution.')
  
    def _default_event_space_bijector(self):
      # TODO(b/145620027) Finalize choice of bijector (consider one that
      # transforms away the heavy tails).
      return identity_bijector.Identity(validate_args=self.validate_args)
  
    def _parameter_control_dependencies(self, is_init):
      if not self.validate_args:
        return []
      assertions = []
      if is_init != tensor_util.is_ref(self.scale):
        assertions.append(assert_util.assert_positive(
            self.scale, message='Argument `scale` must be positive.'))
      return assertions
```

## Numpy

Numpy 中包含 API `numpy.random.Generator.standard_cauchy(size=None)`。

主要参数包括：

- **size：int or tuple of ints, optional.**



Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

- **samples：ndarray or scalar**

The drawn samples.


### 源代码

```python
def standard_cauchy(self, size=None): # real signature unknown; restored from __doc__
    """
    standard_cauchy(size=None)
    
            Draw samples from a standard Cauchy distribution with mode = 0.
    
            Also known as the Lorentz distribution.
    
            Parameters
            ----------
            size : int or tuple of ints, optional
                Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                ``m * n * k`` samples are drawn.  Default is None, in which case a
                single value is returned.
    
            Returns
            -------
            samples : ndarray or scalar
                The drawn samples.
    
            Notes
            -----
            The probability density function for the full Cauchy distribution is
    
            .. math:: P(x; x_0, \gamma) = \frac{1}{\pi \gamma \bigl[ 1+
                      (\frac{x-x_0}{\gamma})^2 \bigr] }
    
            and the Standard Cauchy distribution just sets :math:`x_0=0` and
            :math:`\gamma=1`
    
            The Cauchy distribution arises in the solution to the driven harmonic
            oscillator problem, and also describes spectral line broadening. It
            also describes the distribution of values at which a line tilted at
            a random angle will cut the x axis.
    
            When studying hypothesis tests that assume normality, seeing how the
            tests perform on data from a Cauchy distribution is a good indicator of
            their sensitivity to a heavy-tailed distribution, since the Cauchy looks
            very much like a Gaussian distribution, but with heavier tails.
    
            References
            ----------
            .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
                  Distribution",
                  https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
            .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
                  Wolfram Web Resource.
                  http://mathworld.wolfram.com/CauchyDistribution.html
            .. [3] Wikipedia, "Cauchy distribution"
                  https://en.wikipedia.org/wiki/Cauchy_distribution
    
            Examples
            --------
            Draw samples and plot the distribution:
    
            >>> import matplotlib.pyplot as plt
            >>> s = np.random.default_rng().standard_cauchy(1000000)
            >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
            >>> plt.hist(s, bins=100)
            >>> plt.show()
    """
    pass
```


# 四、对比分析
对第三部分调研的方案进行对比**评价**和**对比分析**，论述各种方案的优劣势。

## 共同点

1. 三者均实现了 Cauchy 分布功能，同时数学原理保持一致，仅是各自实现的方式不同。
2. 三者进行 Cauchy 初始化时，均使用位置向量 loc 以及尺度向量 scale 进行初始化，同时给定了验证参数。
3. pytorch和tensorflow均实现了包括熵（entropy），对数概率密度（log_prob），模式（mode），累积分布密度函数（cdf），以及取样（sample）等方法和属性。
4. pytorch 对均值（mean），方差（variance）进行了处理，会返回 `tensor(nan)`。
5. 三者均实现了对输入参数的检查，虽然各自所实现的方式不同，但总体的功能是类似的。

## 不同点

1. tensorflow 作为更早发展的深度学习框架，相比于其他两者，提供了更多的接口方法，包括但不限于下面：
```python
covariance(
    name='covariance', **kwargs
)
# 协方差仅为非标量时间而定义。
```
```python
cross_entropy(
    other, name='cross_entropy'
)
# 计算（香农）交叉熵。
```
```python
log_cdf(
    value, name='log_cdf', **kwargs
)
# 对数累计分布函数。
```

2. numpy 不同于 pytorch 与 tensorflow，其仅提供了初始化参数需要的条件要求，没有提供Cauchy的众多属性方法，仅是完成在给定size下的分布表示。
3. 三种方式得到的 Cauchy Distribution 的表示不同。比如 numpy 使用 ndarray 或者 scalar 进行表示，pytorch 使用 tensor进行表示。
4. pytorch 支持重参数采样，numpy 与 tensorflow 中仅支持sample，如：
```python
def rsample(self, sample_shape=torch.Size()):
      shape = self._extended_shape(sample_shape)
      eps = self.loc.new(shape).cauchy_()
      return self.loc + eps * self.scale
```

5. numpy 中的 Cauchy 仅是一个方法，并不是一个单独的类，因此没有定义属性和其他和前面两者类似的方法。在使用时，只需要指定 Cauchy 需要的参数即可得到对应的分布，同时也可以再借助 Matplotlib 来展示 Cauchy 分布。

## 方案优点

**pytorch:**
1. 集成的接口比较轻盈，同时实现了 Cauchy Distribution 的核心功能。
2. 源码以及实现思路清晰易懂，api 的使用较为方便。

**tensorflow:**
1. 集成的功能众多，几乎囊括了关于 Cauchy Distribution 的各个方面。

**numpy:**
1. numpy 更多是将 Cauchy Distribution 当作一种工具，需要使用时可以直接使用，没有再为其封装繁琐的方法和属性。

## 方案缺点
1. tensorflow 因集成的功能众多，所以在使用起来比较麻烦，需要花费更多的时间去学习使用。
2. numpy 则因其缺乏 Cauchy Distribution 的其他属性方法，比如 prob / cdf / variance 等，可能会在使用时有缺陷。
3. pytorch 虽然整体简洁轻便，集成了 tensorflow 与 numpy 的各自优势，但是还缺乏一定的额外方法，比如：kl_divergence / log_cdf 等。

# 五、设计思路与实现方案

## 命名与参数设计
直接使用 Cauchy 分布的名称作为此 API 的名称，参数保持 Cauchy 分布最原生的参数即：

- loc：尺度参数
- scale：规模参数

预期 paddle 调用 Cauchy API 的形式为：
```python
#loc: mode or median of the distribution
#scale：half width at half maximum
paddle.distribution.cauchy.Cauchy(loc, scale)
```
## 底层OP设计

使用 paddle 中现存的 API 进行实现，不考虑再令设计底层 OP。

## API实现方案

该 API 在 `paddle.distribution.cauchy` 中实现，部分功能继承父类`Distribution`。
在经过调研对比后，Cauchy API 中设定两个参数：loc 和 scale，其中 loc 和 scale 不同于其他分布，不能单纯的看成分布的均值和方差，而是看作尺度参数和规模参数。

除了 API 调用的基本参数外，`paddle.distribution.cauchy.Cauchy` 中实现的属性、方法主要如下：

### 属性：
- mean ：分布均值
```python
raise NotImplementedError("Cauchy distribution has no mean")
```

- variance：方差
```python
raise NotImplementedError("Cauchy distribution has no variance")
```

- stddev：标准差
```python
raise NotImplementedError("Cauchy distribution has no stddev")
```

> 注：Cauchy 分布的 mean(期望)、variance(方差) 不存在，不存在的原因[1]：
> 
> 1 柯西分布不是一个符合标准正态分布的正态变量;
> 
> 2 柯西分布的概率密度函数无限延申，因此均值和方差无法确定;
> 
> 3 对于柯西分布来说其偏度(skewness)是正的，也就是存在一个右偏分布，导致其分布中的所有值都大于均值,但是偏度度量和均值计算不相关;
> 
> 4 柯西分布属于长尾分布，表示其头尾有极端值，而这些极端值会导致数值求和，当分布倾斜时，极端值会影响期望的计算，所以不存在均值和方差。
> 
> 注：Paddle 为此采取的处理方式：`raise NotImplementedError`, 并阐述原因。


> 参考资料
> 
> [1] 百度文库：柯西分布期望不存在
> 


### 方法
- sample(shape)：随机采样  
在方法内部直接调用本类中的 rsample  方法。(参考pytorch复用重参数化采样结果):
```python
def sample(self, shape):
    rsample(shape)
```

- rsample(shape)：重参数化采样

```python
import paddle
def rsample(self, shape):
    shape = self._extend_shape(shape)
    eps = paddle.standard_normal(shape, dtype=None, name=None)
    cholesky = paddle.linalg.cholesky(self.scale)
  
    return self.loc + batch_mv(cholesky,eps)
```

- prob(value)：概率密度函数

```python
import paddle
def prob(self, value):
    x = paddle.pow(2 * math.pi,-value.shape.pop(1) / 2) * paddle.pow(paddle.linalg.det(self.scale), -1/2)
    y = paddle.exp(-1/2 * paddle.t(value - self.loc) * paddle.inverse(self.scale) * (value - self.loc))
    return x * y
```

- log_prob(value)：对数概率密度函数
```python
import paddle
def log_prob(self, value):
    return paddle.log(self.prob(value))
```

- entropy(value)：熵

```python
import paddle
def entropy(self, value):
    sigma = paddle.linalg.det(self.sclae)
    return 1 / 2 * paddle.log(paddle.pow(2 * math.pi * math.e, value.shpe.pop(1)) * sigma)
```

- kl_divergence 两个Cauchy分布之间的kl散度(other--Cauchy类的一个实例):

```python
import paddle
def kl_divergence(self, other):
    sector_1 = paddle.t(self.loc - other.loc) * paddle.inverse(other.sclae) * (self.loc - other.loc)
    sector_2 = paddle.log(paddle.linalg.det(paddle.inverse(other.sclae) * self.sclae))
    sector_3 = paddle.trace(paddle.inverse(other.scale) * self.sclae)
    n = self.loc.shape.pop(1)
    return 0.5 * (sector_1 - sector_2 + sector_3 - n)
```
在`paddle/distribution/kl.py` 中注册`_kl_cauchy_cauchy`函数，使用时可直接调用`kl_divergence`计算`Cauchy`分布之间的kl散度。

# 六、测试和验收的考量

`test_distribution_cauchy`继承`unittest.TestCase`类中的方法，参考NormalTest的示例，新增一个`CauchyNumpy`类来验证`Cauchy` API的正确性。
- 使用相同的参数实例化 `Cauchy` 类和 `CauchyNumpy` 类，分别调用 `prob`、`log_prob`、`entropy`、`cdf`等方法。将输出的结果进行对比，允许有一定的误差。
- 使用sample方法对多个样本进行测试。

1. 测试 Cauchy 分布的特性

- 测试方法：该部分主要测试分布的均值、方差、熵等特征。类TestCauchy继承unittest.TestCase，分别实现方法setUp（初始化），test_entropy（entropy单测），test_sample（sample单测）。

  * 熵通过Numpy计算相应值，对比Cauchy类中相应property的返回值，若一致即正确；

  * 采样方法除验证其返回的数据类型及数据形状是否合法外，还需证明采样结果符合Cauchy分布。验证策略如下：随机采样30000个cauchy分布下的样本值，计算采样样本的均值和方差，并比较同分布下`scipy.stats.qmc.Cauchy`返回的均值与方差，检查是否在合理误差范围内；同时通过Kolmogorov-Smirnov test进一步验证采样是否属于cauchy分布，若计算所得ks值小于0.1，则拒绝不一致假设，两者属于同一分布；

2. 测试Cauchy分布的概率密度函数

- 测试方法：该部分主要测试分布各种概率密度函数。类TestCauchyPDF继承unittest.TestCase，分别实现方法setUp（初始化），test_prob（prob单测），test_log_prob（log_prob单测），test_cdf（cdf单测）

# 七、可行性分析和排期规划

具体规划为

- 阶段一：完成API功能开发
- 阶段二：完成 `paddle.distribution.cauchy.Cauchy` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `paddle.distribution.cauchy.Cauchy` API，与飞桨2.4代码风格保持一致

# 名词解释

##柯西分布：Cauchy
柯西分布也叫作柯西-劳仑兹分布，它是以奥古斯丁·路易·柯西与亨德里克·劳仑兹名字命名的连续机率分布

> 参考：https://zh.wikipedia.org/zh-tw/%E6%9F%AF%E8%A5%BF%E5%88%86%E5%B8%83 （中文维基百科）
# 附件及参考资料

## PyTorch

[torch.distributions.cauchy.Cauchy](https://pytorch.org/docs/stable/distributions.html?highlight=cauchy#torch.distributions.cauchy.Cauchy)

## TensorFLow

[tfp.distributions.Cauchy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Cauchy#variance)

## Paddle

[paddle.distribution.Beta](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Beta_cn.html)

## Numpy

[numpy.random.Generator.standard_cauchy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_cauchy.html)
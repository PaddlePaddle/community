# paddle.distribution.Gamma 设计文档

|API名称 | paddle.distribution.Gamma | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-28 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230928_api_design_for_distribution_gamma.md<br> | 


# 一、概述
## 1、相关背景
在当前的 Paddle 框架中，`paddle.distribution` 目录内已经实现了一系列概率分布的 API，为了扩展现有的概率分布方案，本次任务计划实现伽马分布（Gamma Distribution）的 API。

Gamma 概率分布的概率密度函数如下：

$$ f(x)=\frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x},(x \geq 0) $$

$$ \Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1} e^{-x} \mathrm{~d} x, (\alpha>0) $$

其中参数 $\alpha$ 称为形状参数，$\beta$ 称为尺度参数。如果一个随机变量 $X$ 呈伽马分布，则可以写作 $X \sim Gamma(\alpha, \beta)$。


## 2、功能目标
为 Paddle 框架增加 `paddle.distribution.Gamma` 的 API，用于伽马分布的概率统计与随机采样。API 包括了如下方法：

- `mean` 计算均值
- `variance` 计算方差 
- `sample` 随机采样
- `rsample` 重参数化采样
- `prob` 概率密度
- `log_prob` 对数概率密度
- `entropy` 熵计算
- `kl` 两个分布间的kl散度

## 3、意义

为 Paddle 增加伽马分布的概率统计与随机采样函数，丰富 Paddle 的概率分布方案，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了 `Distribution` 抽象基类，并且实现了 `Uniform`、`Normal` 等概率分布。目前 Paddle 中暂未实现 `Gamma` 概率分布。


# 三、业内方案调研
## PyTorch

在 PyTorch 中，`Gamma` 概率分布是通过继承 `ExponentialFamily` 类实现。

```python
torch.distributions.Gamma(concentration, rate)
```

使用上面代码可得到形状参数为 `concentration`、尺度参数为 `rate` 的伽马分布。


`Gamma` 类的部分代码如下：
```python
class Gamma(ExponentialFamily):
     r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

        ...

```


## TensorFlow

在 TensorFlow 中，`Gamma` 概率分布是通过继承 `AutoCompositeTensorDistribution` ，这也是概率分布的一般实现方法。

```python
tfp.distributions.Gamma(concentration, rate)
```

使用上面语句可得到形状参数为 `concentration`、尺度参数为 `rate` 的伽马分布。


`Gamma` 类的部分代码如下：
```python
class Gamma(distribution.AutoCompositeTensorDistribution):
  """Gamma distribution.

  The Gamma distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).
  """

  def __init__(self,
               concentration,
               rate=None,
               log_rate=None,
               validate_args=False,
               allow_nan_stats=True,
               force_probs_to_zero_outside_support=False,
               name='Gamma'):
    """Construct Gamma with concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values. Mutually exclusive
        with `log_rate`.
      log_rate: Floating point tensor, natural logarithm of the inverse scale
        params of the distribution(s). Mutually exclusive with `rate`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      force_probs_to_zero_outside_support: If `True`, force `prob(x) == 0` and
        `log_prob(x) == -inf` for values of x outside the distribution support.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    self._force_probs_to_zero_outside_support = (
        force_probs_to_zero_outside_support)
    if (rate is None) == (log_rate is None):
      raise ValueError('Exactly one of `rate` or `log_rate` must be specified.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [concentration, rate, log_rate], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate, dtype=dtype, name='rate')
      self._log_rate = tensor_util.convert_nonref_to_tensor(
          log_rate, dtype=dtype, name='log_rate')

      super(Gamma, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)
```


# 四、对比分析
PyTorch 和 TensorFlow 的 `Gamma` 概率分布分别是继承不同类型的父类实现的。在 PyTorch 中，`Gamma` 类是 `ExponentialFamily` 的子类；在 TensorFlow 中 `Gamma` 类没有依靠其他的概率分布实现，而是同大多数的概率分布一样独立实现。

在 Paddle 中，已经实现了指数族概率分布基类 `ExponentialFamily`，而 `Gamma` 属于指数族概率分布。我们通过继承 `ExponentialFamily` 类来实现 `Gamma` 概率分布。

# 五、设计思路与实现方案

## 命名与参数设计
paddle 调用 `Gamma` 的形式为：

```python
paddle.distribution.Gamma(concentration, rate)
```

`concentration` 和 `rate` 分别是伽马分布的形状参数和尺度参数。


## 底层OP设计
使用 Paddle 现有的 API 可以实现 `Gamma` 概率分布，不涉及底层 OP 的开发。

## API实现方案
### 1、Gamma 类
在目录 `python\paddle\distribution` 下实现 `Gamma` 类，代码的结果如下：
```python
class Gamma(ExponentialFamily):
    def __init__(self, rate):
        self.rate = rate
        super().__init__(self.rate.shape)
        ...

    @property
    def mean(self):
        ...
    
    @property
    def variance(self):
        ...

    def prob(self, value):
        ...

    def log_prob(self, value):
        ...

    def sample(self, shape=()):
        ...

    def entropy(self):
        ...

    @property
    def _natural_parameters(self):
        ...

    def _log_normalizer(self):
        ...

```

### 2、KL散度
在文件 `python\paddle\distribution\kl.py` 中注册伽马分布 KL 散度计算函数。

```python
@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
        ...
```

# 六、测试和验收的考量

测试以 `scipy.stats.gamma` 为基准，测试的主要内容如下：
1. 测试 `mean`、`variance`、`prob`、`log_prob`、`entropy`的准确性，测试参数包含一维和多维参数。

2. 测试 `sample` 和 `rsample` 采样结果的形状、均值和方差是否符合要求，然后使用 Kolmogorov-Smirnov test 验证采样结果是否符合分布要求。此外，还需测试 `rsample` 的采样结果是否支持方向传播。

3. 测试覆盖动态图和静态图模式，覆盖 raise 语句。


# 七、可行性分析和排期规划
## 可行性分析
paddle 中已实现了概率分布的基类 `Distribution`，以及指数族概率分布基类 `ExponentialFamily` ，通过现有的 API，我们可以实现 `Gamma` 概率分布。

## 排期规划
1. 10月1日~10月7日完成 API 开发与调试。

2. 10月8日~10月15日完成测试代码的开发。

# 八、影响面
本次任务涉及以下内容：
1. 新增 `paddle.distribution.Gamma` 模块。
2. 拓展 `paddle.distribution.kl` 模块。

对其他模块无影响。

# 名词解释
无

# 附件及参考资料

1. [PyTorch 的 Gamma 实现](https://github.com/pytorch/pytorch/blob/main/torch/distributions/gamma.py)

2. [TensorFlow 的 Gamma 实现](https://github.com/tensorflow/probability/blob/v0.21.0/tensorflow_probability/python/distributions/gamma.py#L47-L388)

3. [scipy.stats.gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma)
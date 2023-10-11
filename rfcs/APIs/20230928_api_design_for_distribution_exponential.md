# paddle.distribution.Exponential 设计文档

|API名称 | paddle.distribution.Exponential | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | MayYouBeProsperous | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-28 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230928_api_design_for_distribution_exponential.md<br> | 


# 一、概述
## 1、相关背景
在当前的 Paddle 框架中，`paddle.distribution` 目录内已经实现了一系列概率分布的 API，为了扩展现有的概率分布方案，本次任务计划实现指数分布（Exponential Distribution）的 API。

Exponential 概率分布的概率密度函数如下：

$$ f_{F}(x; \theta) =  \theta e^{- \theta x },  (x \ge 0) $$

其中 $\theta > 0$ 是分布的一个参数，常被称为率参数（rate parameter）。如果一个随机变量 $X$ 呈指数分布，则可以写作 $X \sim Exponential(λ)$。

## 2、功能目标
为 Paddle 框架增加 `paddle.distribution.Exponential` 的 API，用于指数分布的概率统计与随机采样。API 包括了如下方法：

- `mean` 计算均值
- `variance` 计算方差 
- `sample` 随机采样
- `rsample` 重参数化采样
- `prob` 概率密度
- `log_prob` 对数概率密度
- `entropy` 熵计算
- `kl` 两个分布间的kl散度

## 3、意义

为 Paddle 增加指数分布的概率统计与随机采样函数，丰富 Paddle 的概率分布方案，进一步完善 Paddle 框架。

# 二、飞桨现状
Paddle 框架内定义了 `Distribution` 抽象基类，并且实现了 `Uniform`、`Normal` 等概率分布。目前 Paddle 中暂未实现 `Exponential` 概率分布。


# 三、业内方案调研
## PyTorch

在 PyTorch 中，`Exponential` 概率分布是通过继承 `ExponentialFamily` 类实现。

```python
torch.distributions.exponential.Exponential(rate) 
```

使用上面语句可得到率参数为 `rate` 的指数分布。


`Exponential` 类的部分代码如下：
```python
class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    def __init__(self, rate, validate_args=None):
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    ...

```


## TensorFlow

在 TensorFlow 中，`Exponential` 概率分布是通过继承 `Gamma` 概率分布实现。

```python
tfp.distributions.Exponential(rate)
```

使用上面代码可得到率参数为 `rate` 的指数分布，分布可写作 $X \sim Exponential (rate)$ 。

`Exponential` 概率分布与 `Gamma` 分布有以下关系，因而 TensorFlow 是使用 `Gamma` 概率分布来实现 `Exponential` 概率分布。

```python
Exponential(rate) = Gamma(concentration=1., rate)
```

`Exponential` 类的部分代码如下：
```python
class Exponential(gamma.Gamma):
  """Exponential distribution.

  The Exponential distribution is parameterized by an event `rate` parameter.
  """

  def __init__(self,
               rate,
               force_probs_to_zero_outside_support=False,
               validate_args=False,
               allow_nan_stats=True,
               name='Exponential'):
    """Construct Exponential distribution with parameter `rate`.
    """
    parameters = dict(locals())
 
    with tf.name_scope(name) as name:
      self._rate = tensor_util.convert_nonref_to_tensor(
          rate,
          name='rate',
          dtype=dtype_util.common_dtype([rate], dtype_hint=tf.float32))
      super(Exponential, self).__init__(
          concentration=1.,
          rate=self._rate,
          allow_nan_stats=allow_nan_stats,
          validate_args=validate_args,
          force_probs_to_zero_outside_support=(
              force_probs_to_zero_outside_support),
          name=name)
      self._parameters = parameters

```


# 四、对比分析
PyTorch 和 TensorFlow 的 `Exponential` 类分别是继承不同类型的父类实现的。在 PyTorch 中，`Exponential` 类和 `Gamma` 类均是 `ExponentialFamily` 的子类，而在 TensorFlow 中 `Exponential` 类是 `Gamma` 类的子类。

在 Paddle 中，已经实现了指数族概率分布基类 `ExponentialFamily`，而 `Exponential` 属于指数族概率分布。我们通过继承 `ExponentialFamily` 类来实现 `Exponential` 概率分布。

# 五、设计思路与实现方案

## 命名与参数设计
paddle 调用 `Exponential` 的形式为：

```python
paddle.distribution.Exponential(rate)
```

`rate` 为指数分布的率参数。


## 底层OP设计
使用 Paddle 现有的 API 可以实现 `Exponential` 概率分布，不涉及底层 OP 的开发。

## API实现方案
### 1、Exponential 类
在目录 `python\paddle\distribution` 下实现 `Exponential` 类，代码的结果如下：
```python
class Exponential(ExponentialFamily):
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
在文件 `python\paddle\distribution\kl.py`中注册指数分布 KL 散度计算函数。

```python
@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
        ...
```

# 六、测试和验收的考量

测试以 `scipy.stats.expon` 为基准，测试的主要内容如下：
1. 测试 `mean`、`variance`、`prob`、`log_prob`、`entropy`的准确性，测试参数包含一维和多维参数。

2. 测试 `sample` 和 `rsample` 采样结果的形状、均值和方差是否符合要求，然后使用 Kolmogorov-Smirnov test 验证采样结果是否符合分布要求。此外，还需测试 `rsample` 的采样结果是否支持方向传播。

3. 测试覆盖动态图和静态图模式，覆盖 raise 语句。


# 七、可行性分析和排期规划
## 可行性分析
paddle 中已实现了概率分布的基类 `Distribution`，以及指数族概率分布基类 `ExponentialFamily` ，通过现有的 API，我们可以实现 `Exponential` 概率分布。

## 排期规划
1. 10月1日~10月7日完成 API 开发与调试。

2. 10月8日~10月15日完成测试代码的开发。

# 八、影响面
本次任务涉及以下内容：
1. 新增 `paddle.distribution.Exponential` 模块。
2. 拓展 `paddle.distribution.kl` 模块。

对其他模块无影响。

# 名词解释
无

# 附件及参考资料

1. [PyTorch 的 Exponential 实现](https://github.com/pytorch/pytorch/blob/main/torch/distributions/exponential.py)

2. [TensorFlow 的 Exponential 实现](https://github.com/tensorflow/probability/blob/v0.21.0/tensorflow_probability/python/distributions/exponential.py#L36-L176)

3. [scipy.stats.expon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html)
# paddle.distribution.Bernoulli 设计文档

|API名称 | 新增API名称 |
|---|---|
|提交作者 | 柳顺(megemini) |
|提交时间 | 2023-03-13 |
|版本号 | V1.0.0 |
|依赖飞桨版本 | V2.4.1 |
|文件名 | 20230313_api_design_for_distribution_bernoulli.md |


# 一、概述
## 1、相关背景
在概率论和统计学中，伯努利分布（`Bernoulli distribution` 以瑞士数学家 `雅各布·伯努利` 命名）是一种随机变量的离散概率分布，取值1为概率 $p$ ，取值0为概率 $q=1-p$。

伯努利分布是 $n=1$ 时（只进行一次实验的）二项分布（`Binomial distribution`）的一种特殊情况。更一般的讲，当 $k=2, n=1$ 时，多项分布（`Multinomial distribution`）即为伯努利分布。

目前的 `Paddle` 框架中暂无 `Bernoulli distribution` 的相关API，特在此任务中结合现有设计方案（如 `Multinomial` 等分布），实现此接口。设计接口为：
- `paddle.distribution.Bernoulli`

## 2、功能目标
为 `paddle` 框架增加API: `paddle.distribution.Bernoulli`，用于 `Bernoulli` 分布的概率统计与随机采样，至少包括如下方法：
- `mean` 计算均值；
- `variance` 计算方差 ；
- `sample` 随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob` 对数概率密度；
- `entropy` 熵计算；
- `kl` 散度计算 (python/paddle/distribution/kl.py)

## 3、意义
完善 `paddle.distribution` 目录下的概率分布相关API，增加 `Bernoulli distribution` 相关接口。

# 二、飞桨现状
## 2.1 paddle.distribution 相关
目前 `paddle` 框架在 `paddle.distribution` 目录下没有 `Bernoulli` 的相关实现。但是，`paddle.distribution` 目录下已经实现了 `Multinomial`（详见 [paddle.distribution.Multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/Multinomial_cn.html#paddle.distribution.Multinomial) ）等接口。
以目前的实现方案，如果需要 `Bernoulli` 的相关能力，可以使用 `paddle.distribution.Multinomial` 进行代替，如:

``` python
    import paddle
    import paddle.distribution

    # Multinomial 的 total_count=1
    # 这里假设取1的概率为0.7，则取零的概率为0.3。
    multinomial = paddle.distribution.Multinomial(1, paddle.to_tensor([0.3, 0.7]))
    print(multinomial.sample((5, )))
    # Tensor(shape=[5, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #     [[0., 1.],
    #      [1., 0.],
    #      [1., 0.],
    #      [0., 1.],
    #      [0., 1.]])

    # 这里采样5000次，可以看到最终采样结果与设定一致。
    a = multinomial.sample((5000, ))
    print(a[:, 0].sum()/5000, a[:, 1].sum()/5000)
    # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    #     [0.29440001]) Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    #     [0.70559996])

```

这种方式虽然可以实现 `Bernoulli` 的相关特性，但是却不够直观，也与目前业内的实现方案不一致，所以有必要单独实现相关API。

另外，`paddle.distribution` 目录下单独实现了 [kl_divergence](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distribution/kl_divergence_cn.html#kl-divergence) 接口，作为KL散度计算的直接对外接口，其内部的完善程度依赖各个分布类的实现，如：

``` python
@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    return p.kl_divergence(q)
```

由此，也需要单独实现 `Bernoulli` 相关API。

## 2.2 tensor.random 相关
`paddle` 框架在 `tensor.random` 目录下实现了若干数学分布相关的接口，如[paddle.bernoulli(x, name=None)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bernoulli_cn.html#paddle.bernoulli) 实现了抽样操作。这些接口仅作为单独的方法对外提供出来，仍需要 `Bernoulli` 等相关分布类提供更完善的特性。

# 三、业内方案调研
目前主流的深度学习框架，如 `PyTorch` 和 `TensorFlow` 都有 `Bernoulli distribution` 的相关接口，其他的开源框架，如 `Numpy` 和 `SciPy` 也有类似的实现。

## 3.1 PyTorch的实现方案
### 3.1.1 相关代码
文档: [Bernoulli](https://pytorch.org/docs/stable/distributions.html#bernoulli)

代码: [torch.distributions.bernoulli.Bernoulli](https://github.com/pytorch/pytorch/blob/master/torch/distributions/bernoulli.py)

### 3.1.2 继承关系
继承自 `ExponentialFamily`
- 文档: [ExponentialFamily](https://pytorch.org/docs/stable/distributions.html#torch.distributions.exp_family.ExponentialFamily)
- 代码: [torch.distributions.exp_family.ExponentialFamily](https://github.com/pytorch/pytorch/blob/master/torch/distributions/exp_family.py)

`PyTorch` 实现 `Bernoulli distribution` 的基类为 `ExponentialFamily`，这是基于伯努利分布属于 `ExponentialFamily` 的数学关系进行实现的。

具体继承关系中，实现了:
- `_natural_params`
- `_log_normalizer`
- `_mean_carrier_measure`

这三个方法/属性，重写了方法:
- `entropy`

### 3.1.3 初始化
实例初始化中主要的参数为:
- `probs` 概率
- `logits`

这里的两个参数 `能且只能` 设置 `一个` 作为初始化参数。

### 3.1.4 主要方法/属性
参考此次需要实现的方法，`PyTorch` 中的具体实现包括:
| 方法 | PyTorch中的实现情况 | 属性(P)/方法(M) |
|---|---|---|
|`mean` 计算均值 | `mean` | P |
|`variance` 计算方差 | `variance` | P |
|`sample` 随机采样 | `sample` | M |
|`rsample` 重参数化采样 | 缺失 | - |
|`prob` 概率密度 | 缺失 | - |
|`log_prob` 对数概率密度 | `log_prob` | M |
|`entropy` 熵计算 | `entropy` | M |
|`cdf` 累计概率密度函数 | 缺失 | - |
|`kl` 两个分布间的kl散度 | 缺失 | - |

### 3.1.5 其他
`PyTorch` 实现了一个 `ContinuousBernoulli`:
- 文档: [ContinuousBernoulli](https://pytorch.org/docs/stable/distributions.html#continuousbernoulli)

- 代码: [torch.distributions.continuous_bernoulli.ContinuousBernoulli](https://github.com/pytorch/pytorch/blob/master/torch/distributions/continuous_bernoulli.py#L12)

在这个分布中实现了 `rsample` 重参数化采样方法。

另外，还实现了 `RelaxedBernoulli`：
- 文档：[RelaxedBernoulli](https://pytorch.org/docs/stable/distributions.html#relaxedbernoulli)

- 代码：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli](https://github.com/pytorch/pytorch/blob/master/torch/distributions/relaxed_bernoulli.py#L94)

同样实现了 `rsample` 重参数化采样方法。

``` python
def rsample(self, sample_shape=torch.Size()):
    shape = self._extended_shape(sample_shape)
    probs = clamp_probs(self.probs.expand(shape))
    uniforms = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
    return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.temperature
```

## 3.2 TensorFlow的实现方案
### 3.2.1 相关代码
文档: [Bernoulli](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Bernoulli)

代码: [tfp.distributions.Bernoulli](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/bernoulli.py)

### 3.2.2 继承关系
继承自
- `DiscreteDistributionMixin`
    - 代码: [tfp.distributions.distribution.DiscreteDistributionMixin](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/distribution.py#L2093)
    - 文档: 无

- `AutoCompositeTensorDistribution`
    - 代码: [tfp.distributions.distribution.AutoCompositeTensorDistribution](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/distribution.py#L2073)
    - 文档: 无

这里生成的文档与代码中不一致，[TensorFlow 的 Bernoulli 文档](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Bernoulli) 中显示此接口继承自:
- `Distribution`
- `AutoCompositeTensor`

但是实际代码中是:
- `DiscreteDistributionMixin`
- `AutoCompositeTensorDistribution`

而其中 `AutoCompositeTensorDistribution` 继承自:
- `Distribution`
- `auto_composite_tensor.AutoCompositeTensor`

这与 `TensorFlow` 的实现方式有关。 `TensorFlow` 中使用 [_DistributionMeta](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/distribution.py#L196) 这个 `metaclass` 来控制（开发建议）各个 `Distribution` 子类的行为（文档的覆盖，方法的实现等），比如: 如果一个子类要调用 `log_prob` 这个方法，则需要实现 `_log_prob(value)` 这个方法。具体可以参考 [_DISTRIBUTION_PUBLIC_METHOD_WRAPPERS](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/distribution.py#L50) 中的映射关系。

``` python
_DISTRIBUTION_PUBLIC_METHOD_WRAPPERS = {
    ...
    'log_prob': '_log_prob',
    ...
}

class _DistributionMeta(abc.ABCMeta):
  """Helper metaclass for tfp.Distribution."""

  def __new__(mcs, classname, baseclasses, attrs):
    ...
    for attr, special_attr in _DISTRIBUTION_PUBLIC_METHOD_WRAPPERS.items():
      if attr in attrs:
        # The method is being overridden, do not update its docstring.
        continue
    ...
```

因此，如果需要扩展 `TensorFlow` 的 `Distribution` 相关分布，则需要先熟悉 `_DistributionMeta` 及相关接口的实现，进而保持实现的一致性。`paddle` 与 `PyTorch` 中的实现没有这么复杂的继承关系与约束，进而扩展起来也相对简单。

### 3.2.3 初始化
实例初始化中主要的参数为:
- `logits`
- `probs` 概率

这里的两个参数 `能且只能` 设置 `一个` 作为初始化参数。

### 3.2.4 主要方法/属性
参考此次需要实现的方法，`TensorFlow` 中的具体实现包括:
| 方法 | TensorFlow中的实现情况 | 属性(P)/方法(M) |
|---|---|---|
|`mean` 计算均值 | `_mean` | M |
|`variance` 计算方差 | `_variance` | M |
|`sample` 随机采样 | `_sample_n` | M |
|`rsample` 重参数化采样 | 缺失 | - |
|`prob` 概率密度 | 使用 `_log_prob` 实现 | M |
|`log_prob` 对数概率密度 | `_log_prob` | M |
|`entropy` 熵计算 | `_entropy` | M |
|`cdf` 累计概率密度函数 | `_cdf` | M |
|`kl` 两个分布间的kl散度 | `_kl_bernoulli_bernoulli` | M |

### 3.2.5 其他
`TensorFlow` 实现了一个 `RelaxedBernoulli`:
- 文档: [RelaxedBernoulli](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/RelaxedBernoulli)

- 代码: [tfp.distributions.RelaxedBernoulli](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/relaxed_bernoulli.py#L31)

作为近似 `Bernoulli` 分布的支持重采样的连续分布。


## 3.3 Numpy的实现方案
`Numpy` 数学分布相关的API集中在 [Random sampling (numpy.random)](https://numpy.org/doc/stable/reference/random/index.html#random-sampling-numpy-random) 模块中。

`Numpy` 将各种分布类型作为 `RandomState` 的方法，而不是将每种分布单独作为类来处理，模块中没有 `Bernoulli distribution` 的接口，但是实现了可替代的接口:
- `binomial`
    - 文档: [random.Generator.binomial](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.binomial.html)
    - 代码: [binomial](https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L3352)
- `multinomial`
    - 文档: [random.Generator.multinomial](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html)
    - 代码: [multinomial](https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L4256)

两者的接口类似，主要输入为:
- `n` 实验次数
- `p`/`pvals` 每次实验的概率

输出为:
- `out` 采样

其实现的效果相当于 `sample` 方法。

## 3.4 SciPy的实现方案
### 3.4.1 相关代码
文档: [scipy.stats.bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)

代码: [bernoulli_gen](https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_discrete_distns.py#L112-L174)

### 3.4.2 继承关系
继承自 `binom_gen`
- 文档: [scipy.stats.binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom)
- 代码: [binom_gen](https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_discrete_distns.py#L29-L106)

`SciPy` 数学分布相关继承关系相对清晰，首先最顶部实现了一个:
- `rv_generic` 类

用于实现连续分布与离散分布的相同能力，如 `random_state` 方法。

然后实现了两个类，都继承自这个 `rv_generic` ，用于区分连续分布与离散分布:
- `rv_continuous` 类
- `rv_discrete` 类

这两个类的实现类似于 `TensorFlow` 中 `_DistributionMeta` 这个 `metaclass` 用于控制其子类的具体实现的方式，如:
``` python
    def pmf(self, k, *args, **kwds):
        ...
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        ...
```

也就是说，子类不直接实现 `pmf` 方法，而是实现 `_pmf`方法，当子类（如 `binom_gen` ）调用 `pmf` 方法时，由其父类先做参数转换等操作，然后再交由子类的 `_pmf` 方法处理。

这个类下面再具体实现各类分布，如:
- `binom_gen` 二项分布类

而由于伯努利分布是一种特殊的二项分布，这里 `bernoulli` 伯努利分布又直接继承自 `binom_gen`。

最后梳理一下 `bernoulli` 伯努利分布的继承关系：
- `rv_generic` 随机变量分布父类
- `rv_discrete` 离散随机变量分布子类
- `binom_gen` 二项分布子类
- `bernoulli_gen` 伯努利分布子类
- `bernoulli = bernoulli_gen(b=1, name='bernoulli')` 伯努利分布实例，类似单例模式

`SciPy` 的这种继承关系具有很好的数学逻辑与代码逻辑，坏处是，后续维护扩展需要先理清上下实现关系。

### 3.4.3 初始化
`bernoulli` 以实例的方式存在（`bernoulli = bernoulli_gen(b=1, name='bernoulli')`），也可以传入一个概率值 $p$ 从而 `"frozen"` 此对象，比如：
``` python
    rv = bernoulli(p)
    rv.pmf(x)
```

### 3.4.4 主要方法/属性
参考此次需要实现的方法，`SciPy` 中的具体实现包括:
| 方法 | SciPy中的实现情况 | 属性(P)/方法(M) |
|---|---|---|
|`mean` 计算均值 | `_stats` 中的 `mu` | M |
|`variance` 计算方差 | `_stats` 中的 `var` | M |
|`sample` 随机采样 | `_rvs` | M |
|`rsample` 重参数化采样 | 缺失 | - |
|`prob` 概率密度 | `_pmf` | M |
|`log_prob` 对数概率密度 | `_logpmf` | M |
|`entropy` 熵计算 | `_entropy` | M |
|`cdf` 累计概率密度函数 | `_cdf` | M |
|`kl` 两个分布间的kl散度 | 缺失 | - |


# 四、对比分析
## 4.1 实现的完整性
- `paddle`: 缺少 `Bernoulli` 的相关API，但可以通过 `Multinomial` 实现相应特性。
- `PyTorch`: 有 `Bernoulli` 的相关API，但缺少 `rsample`、`prob`、 `cdf`、`kl` 等接口。
- `TensorFlow`: 有 `Bernoulli` 的相关API，但缺少 `rsample` 接口。
- `Numpy`: 有 `Bernoulli` 的相关API，但只是实现了 `sample` 特性。
- `SciPy`: 有 `Bernoulli` 的相关API，但缺少 `rsample`、`kl` 等接口。

## 4.2 实现的复杂度
- `paddle`: 不涉及。
- `PyTorch`: 单继承，依赖父类的实现。
- `TensorFlow`: 多继承，依赖父类的实现，需要按照契约编码。
- `Numpy`: 单独的方法。
- `SciPy`: 单继承，依赖父类的实现，逻辑清晰。

## 4.3 可拓展性与可维护性
- `paddle`: 没有强依赖。较易拓展。
- `PyTorch`: 没有强依赖。较易拓展。
- `TensorFlow`: 父类的强依赖。代码逻辑复杂。
- `Numpy`: 不涉及。
- `SciPy`: 父类的强依赖。数学逻辑、代码逻辑清晰。


# 五、设计思路与实现方案
## 命名与参数设计
- 类名: `Bernoulli`
- API: `class paddle.distribution.Bernoulli(probs)`

**注意区分**: `probs` 表示概率，`prob` 表示概率密度。

参考: [飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)

## 底层OP设计
不涉及

## API实现方案
### 继承父类: [ExponentialFamily](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/exponential_family.py#L20)

`class paddle.distribution.Bernoulli(probs)`

### 初始化参数:
- `probs` 概率

    `核心代码`
    ``` python
    def __init__(self, probs):
        self.probs = probs
        self.logits = self._probs_to_logits(probs, is_binary=True)

    ```

### 属性
- `mean`

    均值

    `核心代码`
    ``` python
    @property
    def mean(self):
        return self.probs
    ```

- `variance`

    方差

    `核心代码`
    ``` python
    @property
    def variance(self):
        return self.probs * (1 - self.probs)
    ```


### 方法
- `sample(shape)`

    随机采样，生成满足特定形状的样本数据。

    参数:
    - shape (Sequence[int])：采样形状。

    返回:
    - Tensor，样本数据。

    `核心代码`
    ``` python
    def sample(self, shape):
        return paddle.bernoulli(self.probs.expand(shape))
    ```

- `rsample(shape)`

    重参数化采样

    随机采样，生成满足特定形状的样本数据（重参数采样）。

    参数:
    - shape (Sequence[int])：采样形状。

    返回:
    - Tensor，样本数据。

    实现方案:

    目前在 `PyTorch` 与 `TensorFlow` 框架中，均没有实现 `rsample` 重参数采样方法。分析 `PyTorch` 中拥有 `rsample` 方法的分布，包括:
    - `Beta`
    - `Cauchy`
    - `ContinuousBernoulli`
    - `FisherSnedecor`
    - `Gamma`
    - `HalfCauchy`
    - `HalfNormal`
    - `Kumaraswamy`
    - `LogNormal`
    - `LowRankMultivariateNormal`
    - `MultivariateNormal`
    - `Normal`
    - `RelaxedBernoulli`
    - `LogitRelaxedBernoulli`
    - `RelaxedOneHotCategorical`
    - `Uniform`
    - `Wishart`

    这些分布都是连续型分布。

    [《Bernoulli Variational Auto-Encoder in Torch》](https://davidstutz.de/bernoulli-variational-auto-encoder-in-torch/) 这篇文章归纳了连续分布与离散分布的重采样技巧。对于连续分布，可以使用如下的重采样方法：

    $ {
        z_i = g_i(y, \epsilon_i) = \mu_i(y) + \epsilon_i \sigma_i^2(y)
    } $

    这也是目前代码实现中常用的方式。

    而对于离散分布，比如 `Bernoulli` ，则需要类似 `PyTorch` 实现的 `ContinuousBernoulli` 或 `RelaxedBernoulli` 来实现重采样：

    $ {
        z_i = g(y, \epsilon) = \sigma\left(\ln \epsilon - \ln (1 - \epsilon) + \ln \theta_i(y) - \ln (1 - \theta_i(y))\right)
    } $

    参考 `PyTorch` 中 `RelaxedBernoulli` 的实现，以及 `《Bernoulli Variational Auto-Encoder in Torch》` 的实现方式，这里使用 `Numpy` 实现一个重采样方法：

    `Python Numpy 实现方式`
    ``` python
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def rsample(p, temperature):
        u = np.random.rand()
        return sigmoid((np.log(u)-np.log1p(-u)+np.log(p)-np.log1p(-p))/temperature)

    p = 0.7
    print(np.sum([rsample(p, 0.1)for _ in range(10000)]))
    # 7000.379060342189
    print(np.sum([rsample(p, 1)for _ in range(10000)]))
    # 6388.6837872758815
    print(np.sum([rsample(p, 10)for _ in range(10000)]))
    # 5204.314980195557
    ```

    这里可以看到，随着 `temperature` 趋于 `0` ，重采样方法越近似于 `Bernoulli` 的采样结果。由于目前 `paddle` 没有实现 `RelaxedBernoulli` 等分布的计划，这里实现 `Bernoulli` 的 `rsample` 方法，并默认设置 `temperature` 为 `1` 。另外，需要注意的是，参考 `PyTorch` 与 `《Bernoulli Variational Auto-Encoder in Torch》` 中的实现方式，此 `rsample` 后面需要单独接一个 `Sigmoid activation` 。

    `核心代码`
    ``` python
    def rsample(self, shape):
        temperature = 1
        probs = self.probs.expand(shape)
        uniforms = paddle.rand(shape, dtype=self.probs.dtype)
        return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature

    ```

    **建议**:

    - 参考 `PyTorch` 的实现方式，建议不在 `Bernoulli` 中实现此 `rsample` 接口，抛出 `NotImplementedError` ，后续单独实现类似 `RelaxedBernoulli` 类来实现重采样。
    - 建议后续统一在分布中增加 `is_discrete` 或类似属性，用以区分分布类型，并将 `rsample` 统一在 `is_discrete=False` 的类中实现。
    - 如果需要在此接口中实现重采样，建议增加 `temperature` 参数，而不是目前参考其他 `rsample` 接口只有一个 `shape` 参数。

- `prob(value)`

    概率密度。

    $ { \begin{cases}
        q=1-p & \text{if }value=0 \\
        p & \text{if }value=1
        \end{cases}
    } $

    上述数学公式中 $p$ 表示取值1的概率，取值0为概率 $q=1-p$

    参数:
    - value (Tensor) - 待计算值。

    返回:
    - Tensor，value 对应 `Bernoulli` 概率密度下的值。

    `核心代码` *继承父类*
    ``` python
    def prob(self, value):
        return self.log_prob(value).exp()
    ```

- `log_prob(value)`

    对数概率密度。

    参数:
    - value (Tensor) - 待计算值。

    返回:
    - Tensor，value 对应 `Bernoulli` 对数概率密度值。

    `核心代码`
    ``` python
    def log_prob(self, value):
        logits, value = paddle.broadcast_tensors(
            [self.logits, value]
        )
        return -binary_cross_entropy_with_logits(logits, value, reduction='none')
    ```

- `entropy()`

    信息熵

    $ {
        entropy = -(q \log q + p \log p)
    } $


    上述数学公式中 $p$ 表示取值1的概率，取值0为概率 $q=1-p$

    返回:
    - Tensor，`Bernoulli` 分布的信息熵。

    `核心代码`
    ``` python
    def entropy(self):
        return binary_cross_entropy_with_logits(self.logits, self.probs, reduction='none')
    ```

- `cdf(value)`

    累计概率密度函数。

    ${ \begin{cases}
    0 & \text{if } value \lt  0 \\
    1 - p & \text{if } 0 \leq value \lt  1 \\
    1 & \text{if } value \geq 1
    \end{cases}
    }$

    上述数学公式中 $p$ 表示取值1的概率，取值0为概率 $q=1-p$

    参数:
    - value (Tensor) - 待计算值。

    返回:
    - Tensor，value 对应 `Bernoulli` 累积分布函数下的值。

    `核心代码`
    ``` python
    def cdf(self, value):
        zeros = paddle.zeros_like(self.probs)
        ones = paddle.ones_like(self.probs)
        return paddle.where(value < 0, zeros, paddle.where(value < 1, 1 - self.probs, ones))
    ```

- `kl_divergence(other)`

    两个 `Bernoulli` 分布间的kl散度。

    $ {
        KL(a || b) = p_a \log(p_a / p_b) + (1 - p_a) \log((1 - p_a) / (1 - p_b))
    } $

    参数:
    - other (Bernoulli) - `Bernoulli` 的实例。

    返回:
    - Tensor，两个 `Bernoulli` 分布之间的 KL 散度。

    `核心代码`
    ``` python
    def kl_divergence(self, other):
        a_logits = self.logits
        b_logits = other.logits

        one_minus_pa, pa, log_one_minus_pa, log_pa = _probs_and_log_probs(
            logits=a_logits)
        log_one_minus_pb, log_pb = _probs_and_log_probs(logits=b_logits,
                                                        return_probs=False)

        return (
            (log_pa * pa) -
            (log_pb * pa) +
            (log_one_minus_pa * one_minus_pa) -
            (log_one_minus_pb * one_minus_pa)
        )
    ```

## 其他注意事项
- 初始化参数的校验。
- `0`、`log(0)`、`nan` 等异常值以及相关运算的处理。
- `probs` 的取值范围，在使用 `log` 等相关运算时需要 `clamp` ，避免 `log(0)` 。
- 浮点类型与 `tensor` 的类型转换。
- `broadcast_tensors` 不合理的情况。

# 六、测试和验收的考量
在 `python/paddle/fluid/tests/unittests/distribution` 目录中新增 `test_distribution_bernoulli.py` 文件，用以对 `paddle.distribution.Bernoulli` 接口进行单元测试。

测试基础件：
- `BernoulliNumpy(DistributionNumpy)`

    利用 `Numpy` 和 `Scipy` 构造一个 `Bernoulli` 分布类，预期产生与 `paddle.distribution.Bernoulli` 接口相同且正确的结果。

- `BernoulliTest(unittest.TestCase)`

    参考 [CategoricalTest](https://github.com/PaddlePaddle/Paddle/blob/da551b2e05c8d5fbf6158d1487212386e9be8baf/python/paddle/fluid/tests/unittests/distribution/test_distribution_categorical.py#L55) 测试类的父类，定义接口：

    - `init_dynamic_data` 初始化动态参数
    - `init_static_data` 初始化静态参数

- `BernoulliTestXXX`

    各类单元测试的入口类。

    如：`BernoulliTestError`，表示异常测试。

测试环境：
- 需要在 CPU、GPU 两种场景分别测试。
- 需要固定随机数种子。

验收标准：
- 所有功能测试通过。
- 代码行覆盖率达到 90%以上。
- 测试执行不允许超过 15s。

## 6.1 测试基础特性
测试类：`BernoulliTestFeature(BernoulliTest)`

需要覆盖接口：
- `mean` 计算均值，预期与 `BernoulliNumpy.mean` 一致。
- `variance` 计算方差，预期与 `BernoulliNumpy.variance` 一致。
- `sample` 随机采样，利用 `Kolmogorov-Smirnov test` 预期与 `BernoulliNumpy.sample` 的分布一致。
    ``` python
    In [42]: scipy.stats.kstest([1, 0, 0]*30, scipy.stats.bernoulli(0.3).rvs(100))
    Out[42]: KstestResult(statistic=0.013333333333333334, pvalue=1.0)
    # pvalue>0.05，分布为bernoulli

    In [43]: scipy.stats.kstest([1, 0, 0]*30, scipy.stats.bernoulli(0.7).rvs(100))
    Out[43]: KstestResult(statistic=0.4266666666666667, pvalue=2.8522659012431006e-08)
    # pvalue<0.05，分布不是bernoulli

    ```

- `rsample` 重参数化采样，预期其均值与 `BernoulliNumpy.sample` 在相同的采样数量下的均值，两者差值小于阈值。
- `prob` 概率密度，预期与 `BernoulliNumpy.prob` 一致。
- `log_prob` 对数概率密度，预期与 `BernoulliNumpy.log_prob` 一致。
- `entropy` 熵计算，预期与 `BernoulliNumpy.entropy` 一致。
- `cdf` 累计概率密度函数，预期与 `BernoulliNumpy.cdf` 一致。
- `kl_divergence` 两个分布间的kl散度，两个 `Bernoulli` 的 `Bernoulli.kl_divergence` 预期与两个 `BernoulliNumpy` 的 `BernoulliNumpy.kl_divergence` 一致。

## 6.2 测试动态图和静态图
测试类：`BernoulliTestFeature(BernoulliTest)`

在此类中实现：
- `test_bernoulli_distribution_dygraph` 动态图测试
- `test_bernoulli_distribution_static` 静态图测试

## 6.3 测试数据类型
- 测试类：`BernoulliTestFloat64(BernoulliTestFeature)`

    测试接口在 `float64` 下的基础特性。

- 测试类：`BernoulliTestFloat32(BernoulliTestFeature)`

    测试接口在 `float32` 下的基础特性。

## 6.4 测试多维采样
- 测试类：`BernoulliTestDim(BernoulliTestFeature)`

    测试接口对于 `2-D`、`3-D` 数据的采样结果。

## 6.5 测试异常
测试类：`BernoulliTestError(unittest.TestCase)`

需要验证错误：
- 初始化类型错误。接收 `Tensor`，不接受 `variable` 。
    - `Tensor`
    - `variable`
- 初始化取值错误。
    - 等于 `0` 错误
    - 等于 `1` 错误
    - 小于 `0` 错误
    - 大于 `1` 错误
    - `nan` 错误
- `shape` 类型错误。
- `kl_divergence` 的输入需要是 `Bernoulli`。

参考: [新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
## 可行性分析
`Bernoulli` 分布的概率统计与随机采样，至少包括如下方法：
- `mean` 计算均值；
- `variance` 计算方差 ；
- `sample` 随机采样；
- `prob` 概率密度；
- `log_prob` 对数概率密度；
- `entropy` 熵计算；
- `kl` 散度计算 (python/paddle/distribution/kl.py)

可以包括以下方法：
- `rsample` 重参数化采样；

## 排期规划
- 阶段一：完成接口 `paddle.distribution.Bernoulli` 特性开发。

    预计 `3` 个工作日。

- 阶段二：完成单元测试 `test_distribution_bernoulli` 。

    预计 `3` 个工作日。

- 阶段三：测试、调试接口，提交 `PR`。

    预计 `2` 个工作日。

- 阶段四：完成该 `API` 中英文档及收尾工作。

    预计 `7` 个工作日。

# 八、影响面
本接口实现了 `paddle.distribution.Bernoulli` 此伯努利分布及其相关特性，完善了 `paddle.distribution` 目录下的分布类。

尚有需进一步讨论：
- 是否在本接口实现 `rsample` 方法？参考 `PyTorch` 与 `TensorFlow` 的实现逻辑，不建议在此增加此方法。
- 是否考虑后续增加 `RelaxedBernoulli` 等近似 `Bernoulli` 的连续分布，以实现 `rsample` 重采样方法。
- 需要后续在 `paddle.distribution.kl_divergence` 接口中注册此分布类。

# 名词解释
- 伯努利分布，[https://handwiki.org/wiki/Bernoulli_distribution](https://handwiki.org/wiki/Bernoulli_distribution)

# 附件及参考资料
- `PyTorch`，[torch.distributions.bernoulli.Bernoulli](https://pytorch.org/docs/stable/distributions.html#bernoulli)

- `TensorFlow`，[Bernoulli](https://tensorflow.google.cn/probability/api_docs/python/tfp/distributions/Bernoulli)

- `Numpy`，[Random sampling (numpy.random)](https://numpy.org/doc/stable/reference/random/index.html#random-sampling-numpy-random)

- `SciPy`，[scipy.stats.bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)

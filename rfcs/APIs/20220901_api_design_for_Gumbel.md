# paddle.distribution.Gumbel 设计文档

| API 名称     | paddle.distribution.Gumbel        |
| ------------ | ---------------------------------- |
| 提交作者     | 李芳钰 |
| 提交时间     | 2022-09-1                         |
| 版本号       | V1.0.0                             |
| 依赖飞桨版本 | V2.3.0                             |
| 文件名       | 20220901_api_design_for_Gumbel.md     |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，支持概率分布API，Paddle需要扩充`paddle.distribution.Gumbel` API。

## 2、功能目标

增加 API `paddle.distribution.Gumbel`，用于耿贝尔分布的概率统计与随机采样, 包括如下方法：

- `mean`计算均值；
- `variance`计算方差；
- `stddev`计算标准偏差
- `sample`随机采样；
- `rsample` 重参数化采样；
- `prob` 概率密度；
- `log_prob`对数概率密度；
- `entropy` 熵计算；

## 3、意义

为 Paddle 增加用于 Gumbel 分布的概率统计与随机采样函数，丰富 `paddle.distribution` 中的 API。

# 二、飞桨现状
目前`paddle.distribution` 缺少 Gumbel 分布的实现，
但已有[paddle.distribution.Normal](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/normal.py), [paddle.distribution.Uniform](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/uniform.py), 和 [paddle.distribution.Multinomial](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/multinomial.py) 等API的实现。

# 三、业内方案调研

## PyTorch

PyTorch 中包含 API `torch.distributions.gumbel.Gumbel(loc, scale, validate_args=None)`。 
其[代码位置](https://pytorch.org/docs/stable/_modules/torch/distributions/gumbel.html#Gumbel).

其中核心代码为：
```python
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

    # Explicitly defining the log probability function for Gumbel due to precision issues

    def log_prob(self, value):
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


    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
```
主要的实现逻辑为：

- 根据输入`loc`类型(dtype)的数值限制，确认基本分布`base_dist`服从的均匀分布的参数。
- 然后设置了四个转换分别是 `ExpTransform().inv`, `AffineTransform(loc=0, scale=-torch.ones_like(self.scale))`,`ExpTransform().inv`, `AffineTransform(loc=loc, scale=-self.scale)` 对基本分布`base_dist`进行转换。如下[torch采样代码](https://github.com/pytorch/pytorch/blob/master/torch/distributions/transformed_distribution.py#L108)所示：
```python
    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x
```
- 其余的方法如 `mean`,`entropy` 等就是正常的数学运算。
- 注意 `euler_constant` 为欧拉-马斯切罗尼常数等于0.57721566490153286060。

## Numpy

Numpy 中有API `numpy.random.gumbel(loc=0.0, scale=1.0, size=None)`

### 源代码
[核心代码](https://github.com/numpy/numpy/blob/main/numpy/random/src/distributions/distributions.c#L484)如下:
```c
double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) {
  double U;

  U = 1.0 - next_double(bitgen_state);
  if (U < 1.0) {
    return loc - scale * log(-log(U));
  }
  /* Reject U == 1.0 and call again to get next value */
  return random_gumbel(bitgen_state, loc, scale);
}
```

# 四、对比分析

- 都能实现对 Gumbel 分布进行采样的功能；
- Pytorch 实现了Gumbel分布类，包含了一下属性：均值、方差、标准差等；
- Numpy 只是提供了一种对 Gumbel 分布进行采样的方法；

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.distribution.Gumbel(loc = 0, scale = 1, name = None)
```
参数类型中, 根据所需API要求的类型对输入参数`loc`, `scale`进行限制。

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

- 该 API 实现于 `paddle.distribution.Gumbel`。基于`paddle.distribution` API基类进行开发。
- 对于以下方法`mean`,`log_prob`,`stddev`,`variance`,`entropy`,`prob`等方法和pytorch进行对齐。
- 对于采样方法`sample`,`rsample`参考pytorch通过基本分布结合转换实现，`paddle.distribution.transform` 中有对应的`ExpTransform` 和 `AffineTransform`转换。


# 六、测试和验收的考量

测试考虑的case如下：
- 满足输入`loc`，`scale`的Gumbel分布的均值，方差，标准差，熵等方法的计算是否与公式一致；
- `sample`和`rsample`采样结果的均值，方差和标准差是否符合；
- 输入参数`loc`为不同类型时, 输出的正确性；
- 检查输入参数`scale`是否满足非负性；
- 错误检查：输入`scale`为负数时,能否正确抛出错误；
- 错误检查：输入`loc`和`scale`维度不一致时，能否正确抛出错误；
- 错误检查：输入`loc`为不支持的类型时，能否正确抛出错误；

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`ExpTransform` 和 `AffineTransform` 已经在 Paddle repo 的 python/paddle/distribution/transform.py [目录中](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distribution/transform.py)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

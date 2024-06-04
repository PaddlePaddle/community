# paddle.nn.FeatureAlphaDropout 设计文档

| API名称      | paddle.nn.FeatureAlphaDropout                    |
| ------------ | ------------------------------------------------ |
| 提交作者     | megemini                                         |
| 提交时间     | 2024-06-04                                       |
| 版本号       | V1.0                                             |
| 依赖飞桨版本 | develop版本                                      |
| 文件名       | 20240604_api_design_for_feature_alpha_dropout.md |


# 一、概述

## 1、相关背景

论文 [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515) 提出了一种新的 Dropout 方法，即 Alpha-dropout，它专门设计用来与自归一化激活函数（如 SELU，即 Scaled Exponential Linear Unit）一起使用，以保持网络的自归一化属性。当 Dropout 的目标为整个 channel 时，则为 `FeatureAlphaDropout` 算法。

> 参考赛题：[NO.8 为 Paddle 新增 FeatureAlphaDropout API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no8-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-featurealphadropout-api)


## 2、功能目标

实现 `paddle.nn.FeatureAlphaDropout` 作为独立的 Layer 调用。
实现 `paddle.nn.functional.feature_alpha_dropout` 作为独立的函数调用。

## 3、意义

丰富 Paddle 的 Dropout 相关的 Layer 与 API。

# 二、飞桨现状

目前 Paddle 已经实现了 `paddle.nn.functional.alpha_dropout` 与 `paddle.nn.AlphaDropout` ，暂无针对整个 channel 的 `feature_alpha_dropout` Dropout 接口。

# 三、业内方案调研

PyTorch 实现了相关接口

- [torch.nn.functional.feature_alpha_dropout](https://pytorch.org/docs/stable/generated/torch.nn.functional.feature_alpha_dropout.html#torch.nn.functional.feature_alpha_dropout)
- [torch.nn.FeatureAlphaDropout](https://pytorch.org/docs/stable/generated/torch.nn.FeatureAlphaDropout.html#torch.nn.FeatureAlphaDropout)

具体实现逻辑为在 `aten/src/ATen/native/Dropout.cpp`

``` c++
template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.sym_numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  noise.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}
```

可以看到，由于 `feature alpha dropout` 为 `alpha dropout` 的一种特殊情况，因此， PyTorch 将两者统一实现在以上的同一个函数中，两者的唯一区别为，`feature alpha dropout` 的 `noise` 是通过 `make_feature_noise` 生成的，而不是 `alpha dropout` 中的一个与输入相同形状的空张量：

``` c++
Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sym_sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  c10::SymDimVector sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (C10_UNUSED const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }
  return input.new_empty_symint(sizes);
}
```

从上面的 `make_feature_noise` 可以看到，此处的空张量为至少 `2` 维的张量。取输入维度的前两个，合并维度 `1` 组成，如：

- 输入的维度为 `[2, 3, 4]`
- 输入的空张量为 `[2, 3, 1]`

又如：

- 输入的维度为 `[2, 3, 4, 5, 6, 7]`
- 输入的空张量为 `[2, 3, 1, 1, 1, 1]`

也就是说，此处只保留前两个维度，后面的所有 channel 都相同，也就是实现了 `feature` 的 Dropout 操作。

# 四、对比分析

Paddle 目前实现了 `alpha_dropout` ，`python/paddle/nn/functional/common.py` 中：

``` python
def alpha_dropout(x, p=0.5, training=True, name=None):
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a float or int")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")

    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'alpha_dropout'
        )

    if training:
        if p == 1:
            return paddle.scale(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p**2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        if not feature_dropout:
            input_shape = x.shape
        else:
            if x.ndim < 2:
                raise ValueError(
                    'Feature alpha dropout needs at least 2D input.'
                )
            input_shape = list(x.shape[:2]) + [1] * len(x.shape[2:])

        # get mask
        random_tensor = paddle.uniform(
            input_shape, dtype='float32', min=0.0, max=1.0
        )
        p = full(shape=input_shape, fill_value=p, dtype='float32')
        keep_mask = paddle.greater_equal(random_tensor, p)
        keep_mask = paddle.cast(keep_mask, dtype)
        drop_mask = paddle.subtract(
            full(shape=input_shape, fill_value=1.0, dtype=dtype), keep_mask
        )

        # apply mask
        b = full(shape=input_shape, fill_value=b, dtype=dtype)
        y = paddle.add(
            paddle.multiply(x, keep_mask),
            paddle.scale(drop_mask, scale=alpha_p),
        )
        res = paddle.add(paddle.scale(y, scale=a), b, name=name)
        return res
    else:  # test
        return x
```

其计算逻辑与 PyTorch 一致，但由于没有对于 `feature` 单独处理，因此，只实现了基础的 `alpha_dropout` 函数。

# 五、设计思路与实现方案

由于 `feature_alpha_dropout` 只是 `alpha_dropout` 的一种特例，因此，可以复用 Paddle 中的 `alpha_dropout` 函数进行实现：

- 实现 `paddle.nn.functional.feature_alpha_dropout` 作为独立的函数调用。
- 实现 `paddle.nn.FeatureAlphaDropout` 调用上面的函数，作为独立的 Layer 调用。

为了复用 `alpha_dropout` 函数，这里可以抽取一个统一的函数 `_feature_alpha_dropout_impl`：

``` python
def _feature_alpha_dropout_impl(
    x: paddle.Tensor,
    feature_dropout: bool,
    p: float | int,
    training: bool = True,
    name: str | None = None,
) -> paddle.Tensor:

```

然后，在 `alpha_dropout` 与 `feature_alpha_dropout` 中调用 `_feature_alpha_dropout_impl`：

``` python
def alpha_dropout(x, p=0.5, training=True, name=None):
    return _feature_alpha_dropout_impl(
        x, feature_dropout=False, p=p, training=training, name=name
    )


def feature_alpha_dropout(x, p=0.5, training=True, name=None):
    return _feature_alpha_dropout_impl(
        x, feature_dropout=True, p=p, training=training, name=name
    )
```

其唯一区别为 `feature_dropout` 的参数不同。

## 命名与参数设计

### 作为函数调用

``` python
def _feature_alpha_dropout_impl(
    x: paddle.Tensor,
    feature_dropout: bool,
    p: float | int,
    training: bool = True,
    name: str | None = None,
) -> paddle.Tensor:

def alpha_dropout(x, p=0.5, training=True, name=None): ...
def feature_alpha_dropout(x, p=0.5, training=True, name=None): ...
```

其中:

- x (Tensor)，输入的张量
- feature_droput (bool)，是否为针对整个 Channel
- p (float | int)，设为零的概率
- training (bool)，是否为训练阶段
- name (str)，名称

### 作为 Layer

``` python
class FeatureAlphaDropout(Layer): ...
```

## 底层OP设计

直接在 Python 层实现，不涉及底层算子。

## API实现方案

参考代码：

``` python
def _feature_alpha_dropout_impl(
    x: paddle.Tensor,
    feature_dropout: bool,
    p: float | int,
    training: bool = True,
    name: str | None = None,
) -> paddle.Tensor:
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a float or int")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")

    if not in_dynamic_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'uint16', 'float32', 'float64'], 'alpha_dropout'
        )

    if training:
        if p == 1:
            return paddle.scale(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p**2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype

        # 此处是与原 `alpha_dropout` 唯一的不同之处
        if not feature_dropout:
            input_shape = x.shape
        else:
            if x.ndim < 2:
                raise ValueError(
                    'Feature alpha dropout needs at least 2D input.'
                )
            input_shape = list(x.shape[:2]) + [1] * len(x.shape[2:])

        # get mask
        random_tensor = paddle.uniform(
            input_shape, dtype='float32', min=0.0, max=1.0
        )
        p = full(shape=input_shape, fill_value=p, dtype='float32')
        keep_mask = paddle.greater_equal(random_tensor, p)
        keep_mask = paddle.cast(keep_mask, dtype)
        drop_mask = paddle.subtract(
            full(shape=input_shape, fill_value=1.0, dtype=dtype), keep_mask
        )

        # apply mask
        b = full(shape=input_shape, fill_value=b, dtype=dtype)
        y = paddle.add(
            paddle.multiply(x, keep_mask),
            paddle.scale(drop_mask, scale=alpha_p),
        )
        res = paddle.add(paddle.scale(y, scale=a), b, name=name)
        return res
    else:  # test
        return x


def alpha_dropout(x, p=0.5, training=True, name=None):
    return _feature_alpha_dropout_impl(
        x, feature_dropout=False, p=p, training=training, name=name
    )


def feature_alpha_dropout(x, p=0.5, training=True, name=None):
    return _feature_alpha_dropout_impl(
        x, feature_dropout=True, p=p, training=training, name=name
    )

```

与原 `alpha_dropout` 唯一不同的地方是 `input_shape` 的不同：

``` python
        # 此处是与原 `alpha_dropout` 唯一的不同之处
        if not feature_dropout:
            input_shape = x.shape
        else:
            if x.ndim < 2:
                raise ValueError(
                    'Feature alpha dropout needs at least 2D input.'
                )
            input_shape = list(x.shape[:2]) + [1] * len(x.shape[2:])

```

# 六、测试和验收的考量

- **编程范式场景**
  - 常规覆盖动态图 (和静态图) 的测试场景。

- **硬件场景**
  - 常规需覆盖 CPU、GPU 两种测试场景。

- **输入参数**
  - 常规覆盖默认参数，常用参数，错误参数。
  - 常规数据类型 float16, float32 or float64

# 七、可行性分析和排期规划

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

丰富 paddle API，对其他模块没有影响

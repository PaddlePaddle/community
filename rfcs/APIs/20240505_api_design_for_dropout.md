# paddle_slice_scatter 设计文档

| API名称                                                      | paddle.nn.FeatureAlphaDropout                  |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | magnetowang                                     |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2024-05-05                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   |
| 文件名                                                       | 20240505_api_design_for_FeatureAlphaDropout.md<br> |

**修订记录**

v1.0 
实现 FeatureAlpha 方式的 Dropout。详情可以参考论文：Self-Normalizing Neural Networks，新增paddle.nn.FeatureAlphaDropout


# 一、概述
## 1、相关背景

为了提升飞桨API丰富度，需要为飞桨扩充API `paddle.nn.FeatureAlphaDropout`

本API属于飞桨开源个人贡献赛API开发任务[NO.8 为 Paddle 新增 FeatureAlphaDropout API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no8-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-featurealphadropout-api)的任务。

## 2、功能目标

实现 FeatureAlphaDropout 函数

预期该API支持

- paddle.nn.FeatureAlphaDropout 作为独立的函数调用

## 3、意义

为飞桨增加新的dropout类型函数，用于复现论文，对齐业界竞品能力

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.nn.FeatureAlphaDropout`
并且是c++和cuda两种实现


其介绍为：

> As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

参数表为：
- Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
- Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

### cpp实现函数
```cpp

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






# 四、对比分析

对比 PyTorch :
- PyTorch 有 cpp 和 cuda 两种实现方法
  




# 五、设计思路与实现方案

paddle 在原来基础上添加 noise， 再复用 alpha_dropout 函数，

## 命名与参数设计

添加 Python API:
```python
paddle.nn.FeatureAlphaDropout(p=0.5)
```

参数表：

- input: (Tensor) 输入的 tensor。维度至少大于2
- p: 概率值，float类型，默认为0.5


## 底层OP设计

不涉及底层 OP。

## API实现方案

此次在 `paddle.nn`模块上 新增算子实现算子
修改文件
Paddle/python/paddle/nn/layer/common.py
Paddle/python/paddle/nn/functional/common.py
测试文件
Paddle/test/legacy_test/test_dropout_op.py

``` python
class FeatureAlphaDropout(Layer):
    def __init__(self, p=0.5, name=None)
    def forward(self, input)
    def extra_repr(self)
    
```

# 六、测试和验收的考量

- 覆盖 CPU 测试场景，暂不考虑GPU场景
- 支持各种Tensor精度，FP32、FP64 等（待验证）
- 需要检查计算正确性
- 需要检查多维的情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关PythonAPI均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

# 名词解释

# 附件及参考资料

[NO.8 为 Paddle 新增 FeatureAlphaDropout API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no8-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-featurealphadropout-api)

[PyTorch FeatureAlphaDropout 文档](https://pytorch.org/docs/stable/generated/torch.nn.FeatureAlphaDropout.html)
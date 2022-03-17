# paddle.nn.ChannelShuffle设计文档

|API名称 | paddle.nn.ChannelShuffle |
|---|---|
|提交作者 | 为往圣继绝学 |
|提交时间 | 2022-03-17 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20200317_api_design_for_ChannelShuffle.md |


# 一、概述
## 1、相关背景
ChannelShuffle操作由论文《[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)》提出，这个操作通过简单的转置来均匀地打乱了通道间的顺序，从而增加了通道间的信息流动，提高了特征的重用率。

## 2、功能目标

在飞桨框架中增加ChannelShuffle 组网 API。

## 3、意义

飞桨将支持ChannelShuffle组网 API。

# 二、飞桨现状
飞桨目前不支持此功能，但可以通过组合API的方式实现此功能：
```python
class ChannelShuffle(paddle.nn.Layer):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        self.data_format = data_format
    def forward(self, x):
        n, c, h, w = x.shape
        x = paddle.reshape(x, [n, self.groups, c / self.groups, h, w])
        x = paddle.transpose(x, [0, 2, 1, 3, 4])
        x = paddle.reshape(x, [n, c, h, w])
        return x
```

# 三、业内方案调研
## PyTorch

PyTorch中已经有了`torch.nn.ChannelShuffle`（文档在[这里](https://pytorch.org/docs/stable/_modules/torch/nn/modules/channelshuffle.html#ChannelShuffle)），但在1.11.0版本中试验后发现其不支持反向传播。在Python层面的主要实现代码为：

```python
class ChannelShuffle(Module):
    def __init__(self, groups: int) -> None:
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        return F.channel_shuffle(input, self.groups)
```

其中的`F.channel_shuffle`是由C++实现的，主要代码为：

```c++
Tensor math_channel_shuffle(const Tensor& self, int64_t groups) {
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});
  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
      .contiguous()
      .reshape(self.sizes());
  return namedinference::propagate_names_if_nonempty(
      output_tensor,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}
```

## TensorFlow

TensorFlow目前没有直接提供`ChannelShuffle`的API，但是也有[网友](https://blog.csdn.net/baidu_23388287/article/details/94456951)通过组合API的方式实现了该操作：

```python
def shuffle_unit(self, x, groups):
    with tf.variable_scope('shuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
```

## 实现逻辑

我们以NCHW格式的4D张量为例来描述ChannelShuffle的实现逻辑：
1. 把形为[N, C, H, W]的张量重塑成[N, g, C/g, H, W]的形状；
2. 对第1维度（N为第0维度）和第2维度进行转置，得到形为[N, C/g, g, H, W]的张量；
3. 把形为[N, C/g, g, H, W]的张量重塑为[N, C, H, W]的形状。

# 四、对比分析
- PyTorch在C++层面为ChannelShuffle设计了底层算子，执行效率高，但不利于（不参与框架设计）开发者从源码层面了解该API的行为。
- 使用组合API的方式来实现ChannelShuffle是简洁易懂的，开发者可以直接从Python源码的层面来了解该API的行为，但在执行效率上可能会逊色于原生算子实现方案。

无论是C++实现还是组合API实现，其逻辑都是十分简单的，故考虑使用C++编写新的算子以期取得更高的效率。另一方面，可以在文档中详细描述该API的行为，避免（不参与框架设计的）开发者在阅读源码上耗费过多时间。

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.nn.ChannelShuffle(groups, data_format='NCHW')`，

- **groups** (int) - 要把通道数分成的组数，必需整除通道数。
- **data_format**(str) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

## 底层OP设计
ChannelShuffle与飞桨中已有的PixelShuffle在操作上是类似的，参考PixelShuffle的OP设计来实现ChannelShuffle的底层OP。

核函数的原型（定义在`paddle/phi/kernels/channel_shuffle_kernel.h`）设计为

```c++
template <typename T, typename Context>
void ChannelShuffleKernel(const Context& ctx,
                          const DenseTensor& x,
                          int groups,
                          const std::string& data_format,
                          DenseTensor* out);
```

反向核函数的原型（定义在`paddle/phi/kernels/channel_shuffle_grad_kernel.h`）设计为

```c++
template <typename T, typename Context>
void ChannelShuffleGradKernel(const Context& ctx,
                              const DenseTensor& out_grad,
                              int groups,
                              const std::string& data_format,
                              DenseTensor* x_grad);
```

它们的实现定义在`paddle/phi/kernels/impl/channel_shuffle_kernel_impl.h`和`paddle/phi/kernels/impl/channel_shuffle_grad_kernel_impl.h`，它们的注册放在

- `paddle/phi/kernels/cpu/channel_shuffle_kernel.cc`
- `paddle/phi/kernels/cpu/channel_shuffle_grad_kernel.cc`
- `paddle/phi/kernels/gpu/channel_shuffle_kernel.cu`
- `paddle/phi/kernels/gpu/channel_shuffle_grad_kernel.cu`

## API实现方案

参考`paddle.nn.PixelShuffle`来实现`paddle.nn.ChannelShuffle`，顺便实现`paddle.nn.functional.channel_shuffle`。

### nn.functional.channel_shuffle

将`nn.functional.channel_shuffle`定义在`python/paddle/nn/functional/vision.py`中：

```python
def channel_shuffle(x, groups, data_format="NCHW", name=None):
    # ...
    if in_dynamic_mode():
        return _C_ops.channel_shuffle(x, "groups", groups,
                                      "data_format", data_format)
    # ...
```

### nn.ChannelShuffle

将`nn.ChannelShuffle`定义在`python/paddle/nn/layer/vision.py`中：

```python
class ChannelShuffle(Layer):
    def __init__(self, groups, data_format="NCHW", name=None):
        pass

    def forward(self, x):
        return functional.channel_shuffle(x, self._groups,
                                          self._data_format, self._name)

    def extra_repr(self):
        pass
```

# 六、测试和验收的考量

- 是否同时支持静态图和动态图；
- 是否同时支持CPU和GPU平台；
- 测试不同张量类型下的表现；
- 对全部入参进行参数有效性和边界值测试，确定每个入参都可以正确生效；
- 前向计算的正确性（与组合API实现比较、与PyTorch比较）；
- 反向计算的正确性；
- 当传入的`group`不合法（不是正整数、不整除通道数）时会抛出异常并有友好的提示。


# 七、可行性分析和排期规划
已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面
`ChannelShuffle`为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无

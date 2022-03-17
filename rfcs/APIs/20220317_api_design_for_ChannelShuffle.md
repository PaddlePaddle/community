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

飞桨将支持ChannelShuffle 组网 API。

# 二、飞桨现状
飞桨目前不支持此功能，但可以通过组合API的方式实现此功能。


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

# 四、对比分析
无论是C++实现还是组合API实现，其逻辑都是十分简单的，故考虑使用C++编写新的算子以期取得更高的效率。

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.nn.ChannelShuffle(groups, data_format='NCHW')`，

- **groups** (int) - 要把通道数分成的组数，必需整除通道数。
- **data_format**(str) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

## 底层OP设计
ChannelShuffle与飞桨中已有的PixelShuffle在操作上是类似的，参考PixelShuffle的OP设计来实现ChannelShuffle的底层OP。

## API实现方案

参考`paddle.nn.PixelShuffle`来实现`paddle.nn.ChannelShuffle`，顺便实现`paddle.nn.functional.channel_shuffle`。

# 六、测试和验收的考量

考虑测试的情况：
- 与PyTorch的结果的一致性；
- 反向传播的正确性；
- 错误检查：`groups`不合法或不整除通道数时能正确抛出异常。

# 七、可行性分析和排期规划
已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面
`ChannelShuffle`为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无

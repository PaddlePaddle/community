# paddle.nn.PxielUnshuffle设计文档

| API名称      | paddle.nn.PxielUnshuffle                  |
| ------------ | ----------------------------------------- |
| 提交作者     | 为往圣继绝学                              |
| 提交时间     | 2022-03-17                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本 | develop                                   |
| 文件名       | 20220317_api_design_for_PxielUnshuffle.md |


# 一、概述

## 1、相关背景

PixelUnshuffle是PixelShuffle 的逆操作，由论文《[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)》提出。这个操作通过将空间维度的特征转移到通道，来将亚像素卷积的结果恢复。

## 2、功能目标

在飞桨框架中增加PxielUnshuffle组网 API。

## 3、意义

飞桨将支持PxielUnshuffle组网 API。

# 二、飞桨现状

飞桨目前不支持此功能，但可以通过组合API的方式实现此功能：

```python
class PixelUnshuffle(paddle.nn.Layer):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.factor = downscale_factor
        self.data_format = data_format

    def forward(self, x):
        n, c, h, w = x.shape
        x = paddle.reshape(x, [n, c, h / self.factor, self.factor, w / self.factor, self.factor])
        x = paddle.transpose(x, [0, 1, 3, 5, 2, 4])
        x = paddle.reshape(x, [n, c * self.factor * self.factor, h / self.factor, w / self.factor])
        return x
```

# 三、业内方案调研

## PyTorch

PyTorch中已经有了`torch.nn.PixelUnshuffle`（文档在[这里](https://pytorch.org/docs/stable/_modules/torch/nn/modules/pixelshuffle.html#PixelUnshuffle)）。在Python层面的主要实现代码为：

```python
class PixelUnshuffle(Module):
    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_unshuffle(input, self.downscale_factor)
```

其中的`F.pixel_unshuffle`是由C++实现的，主要代码为：

```c++
Tensor pixel_unshuffle(const Tensor& self, int64_t downscale_factor) {
  TORCH_CHECK(self.dim() >= 3,
              "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
              self.dim(), " dimension(s)");
  TORCH_CHECK(
      downscale_factor > 0,
      "pixel_unshuffle expects a positive downscale_factor, but got ",
      downscale_factor);
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  TORCH_CHECK(h % downscale_factor == 0,
             "pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=", h,
             " is not divisible by ", downscale_factor)
  TORCH_CHECK(w % downscale_factor == 0,
             "pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)=", w,
             " is not divisible by ", downscale_factor)
  int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  int64_t oc = c * downscale_factor_squared;
  int64_t oh = h / downscale_factor;
  int64_t ow = w / downscale_factor;

  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {c, oh, downscale_factor, ow, downscale_factor});
  const auto input_reshaped = self.reshape(added_dims_shape);

  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* c */, -3 /* 1st downscale_factor */, -1 /*2nd downscale_factor */,
                                         -4 /* oh */, -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);

  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});
  return input_permuted.reshape(final_shape);
}

}} // namespace at::native

```

PyTorch在C++层面执行步骤为：

1. 检查张量维度（要求>=3）；
2. 检查downscale_factor是否为正数；
3. 检查downscale_factor是否同时整除宽度和高度；
4. 计算输出张量的形状；
5. 按`PixelUnshuffle`的逻辑（见本小节末尾）向前计算。

## TensorFlow

TensorFlow目前没有直接提供`PixelUnshuffle`的API，但是也可以通过组合API的方式实现该操作：

```python
def pixel_unshuffle_unit(self, x, downscale_factor):
    with tf.variable_scope('pixel_unshuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h / downscale_factor, downscale_factor, w / downscale_factor, downscale_factor, c]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 3, 5, 2, 4]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h / downscale_factor, w / downscale_factor, c * downscale_factor * downscale_factor]))

```

## 实现逻辑

我们以NCHW格式的4D张量为例来描述PixelUnshuffle的实现逻辑：

1. 把形为[N, C, H, W]的张量重塑成[N, C, H/r, r, W/r, r]的形状；
2. 转置得到形为[N, C, r, r, H/r, W/r]的张量；
3. 把形为[N, C, r, r, H/r, W/r]的张量重塑为[N, C\*r\*r, H/r, W/r]的形状。

这里的r就是代码中的`downscale_factor`。

# 四、对比分析

在功能目的上，PyTorch的实现和Tensorflow的实现是一致的（其逻辑见上一小节末尾）。

不过PyTorch只支持NCHW格式的张量，而这里的TensorFlow实现是针对NHWC格式的。由于TensorFlow是通过组合API实现的，所以我们也很容易使它再支持NCHW格式的张量。

考虑到PixelUnshuffle一般是结合卷积操作使用的，而飞桨中的卷积有`data_format`参数，因此我们可以让PixelUnshuffle也拥有`data_format`参数。

# 五、设计思路与实现方案

## 命名与参数设计

API设计为`paddle.nn.PixelUnshuffle(downscale_factor, data_format='NCHW')`，

- **downscale_factor** (int) - 空间分辨率缩小的比例，需要同时整除宽度和高度；
- **data_format**(str) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

## 底层OP设计

PixelUnshuffle与飞桨中已有的PixelShuffle操作在逻辑上是一样的，故可参考PixelShuffle的OP设计来实现PixelUnshuffle的底层OP。

核函数的原型（定义在`paddle/phi/kernels/pixel_unshuffle_kernel.h`）设计为

```c++
template <typename T, typename Context>
void PixelUnshuffleKernel(const Context& ctx,
                          const DenseTensor& x,
                          int downscale_factor,
                          const std::string& data_format,
                          DenseTensor* out);
```

其实现方式与设备无关，将其实现以及在CPU、GPU上的注册放在`paddle/phi/kernels/pixel_unshuffle_kernel.cc`中。

反向核函数的原型（定义在`paddle/phi/kernels/pixel_unshuffle_grad_kernel.h`）设计为

```c++
template <typename T, typename Context>
void PixelUnshuffleGradKernel(const Context& ctx,
                            const DenseTensor& out_grad,
                            int downscale_factor,
                            const std::string& data_format,
                            DenseTensor* x_grad);
```

其实现方式与设备无关，将其实现以及在CPU、GPU上的注册放在`paddle/phi/kernels/pixel_unshuffle_grad_kernel.cc`中。

## API实现方案

参考`paddle.nn.PixelShuffle`来实现`paddle.nn.PixelUnshuffle`，顺便实现`paddle.nn.functional.pixel_unshuffle`。

### nn.functional.pixel_unshuffle

将`nn.functional.pixel_unshuffle`定义在`python/paddle/nn/functional/vision.py`中：

```python
def pixel_unshuffle(x, downscale_factor, data_format="NCHW", name=None):
    # ...
    if in_dynamic_mode():
        return _C_ops.pixel_unshuffle(x, "downscale_factor", downscale_factor,
                                      "data_format", data_format)
		# ...
    return out
```

### nn.PixelUnshuffle

将`nn.PixelUnshuffle`定义在`python/paddle/nn/layer/vision.py`中：

```python
class PixelUnshuffle(Layer):
    def __init__(self, downscale_factor, data_format="NCHW", name=None):
        pass

    def forward(self, x):
        return functional.pixel_unshuffle(x, self._downscale_factor,
                                          self._data_format, self._name)

    def extra_repr(self):
        pass
```

# 六、测试和验收的考量

- 错误检查：
  - `downscale_factor`不是整数时抛出异常，
  - `downscale_factor`不是正数时抛出异常，
  - `data_format`不是NCHW和NHWC中的一个时抛出异常，
  - 输入的`x`不是4D张量时抛出异常；
 - 向前计算：`data_format`分别为NCHW和NHWC时与NumPy的一致性
 - 反向计算：`data_format`分别为NCHW和NHWC时与NumPy的一致性
 - 平台支持：CPU和GPU
 - 支持静态图和动态图

# 七、可行性分析和排期规划

已经实现，待该设计文档通过验收后可马上提交。

# 八、影响面

`PxielUnshuffle`为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无
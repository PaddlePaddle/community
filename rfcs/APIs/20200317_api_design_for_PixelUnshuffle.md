# paddle.nn.PixelUnshuffle设计文档

|API名称 | paddle.nn.PixelUnshuffle |
|---|---|
|提交作者 | thunder95 |
|提交时间 | 2022-03-17 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20200317_api_design_for_PixelUnshuffle.md |


# 一、概述
## 1、相关背景
PixelUnshuffle 是 PixelShuffle 的逆操作，通过将空间维度的特征转移到通道，将亚像素卷积的结果恢复, 可参考论文《[Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158)》。

## 2、功能目标

在飞桨框架中增加PixelUnshuffle 组网 API。

## 3、意义

飞桨将支持PixelUnshuffle 组网 API。

# 二、飞桨现状
飞桨目前不支持此功能，但可以通过三个步骤组合内置API的方式实现此功能。

```python
n, c, h, w = x.shape
downscale_factor = 4
x = paddle.reshape(x, [n, c, h // downscale_factor, downscale_factor, w // downscale_factor, downscale_factor])
x = paddle.transpose(x, [0, 1, 3, 5, 2, 4])
x = paddle.reshape(x, [n, c * downscale_factor * downscale_factor, h // downscale_factor, w // downscale_factor])
```

# 三、业内方案调研
## PyTorch

PyTorch中已经有了`torch.nn.PixelUnshuffle`（文档在[这里](https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html?highlight=pixelunshuffle#torch.nn.PixelUnshuffle)）。在Python层面的主要实现代码为：

```python
class PixelUnshuffle(Module):
        __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)
```

其中的`F.pixel_unshuffle`是由C++实现的，主要代码为：

```c++
Tensor pixel_unshuffle(const Tensor& self, int64_t downscale_factor) {
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;
  int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  int64_t oc = c * downscale_factor_squared;
  int64_t oh = h / downscale_factor;
  int64_t ow = w / downscale_factor;

  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {c, oh, downscale_factor, ow, downscale_factor});
  const auto input_reshaped = self.reshape(added_dims_shape);
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* c */, -3 /* 1st downscale_factor */, -1 /*2nd downscale_factor */,
                                         -4 /* oh */, -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});
  return input_permuted.reshape(final_shape);
}
```

TensorFlow目前也没有直接提供PixelUnshuffle的API，但是也可以通过基础API进行组网实现:

def pixel_unshuffle_unit(self, x, downscale_factor):
    with tf.variable_scope('pixel_unshuffle_unit'):
        n, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h / downscale_factor, downscale_factor, w / downscale_factor, downscale_factor, c]))
        x = tf.transpose(x, tf.convert_to_tensor([0, 1, 3, 5, 2, 4]))
        x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h / downscale_factor, w / downscale_factor, c * downscale_factor * downscale_factor]))

主要逻辑步骤包括:
1. 先改变张量形状,[N, C, H, W]为[N, C, H/factor, factor, W/factor, factor]；
2. 通过转置操作, 得到形为[N, C, factor, factor, H/factor, W/factor]的张量；
3. 再将第二步得到结果改为[N, C*factor*factor, H/factor, W/factor]的形状。


# 四、对比分析
目前可以三种方案实现，组合API，C++ CPU算子， CUDA GPU算子。 组合API的方式的执行效率以及显存效率会稍微差一些; CUDA GPU算子可以将全部运算逻辑封装在一个kernel下运行，但就在这种简单逻辑处理来讲， 效率不会有非常大的提升。
Pytorch支持三维张量, 对于四维张量来说相当于一个单通道，所以优势并不突出
Pytorch不支持NHWC格式输入
无论是C++实现还是组合API实现，其逻辑都是十分简单的，故考虑使用C++编写新的算子以期取得更高的效率。


# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.nn.PixelUnshuffle(downscale_factor, data_format="NCHW", name=None)`，

- **downscale_factor** (int) - 长宽缩放的倍数因子，必需整除长宽维度。
- **data_format**(str) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。

## 底层OP设计
PixelUnshuffle与飞桨中已有的PixelShuffle在操作上是类似的，参考PixelShuffle的OP设计来实现PixelUnshuffle的底层OP。

## API实现方案

参考`paddle.nn.PixelShuffle`来实现`paddle.nn.PixelUnshuffle`，顺便实现`paddle.nn.functional.pixel_unshuffle`。

## 代码实现文件路径

在文件paddle/phi/kernels/impl/pixel_unshuffle_kernel_impl.h和paddle/phi/kernels/impl/pixel_unshuffle_grad_kernel_impl.h中编写主要的正向和反向计算逻辑：


```c++
template <typename T, typename Context>
void PixelUnshuffleKernel(const Context& ctx,
                          const DenseTensor& x,
                          int downscale_factor,
                          const std::string& data_format,
                          DenseTensor* out);
                          
template <typename T, typename Context>
void PixelUnshuffleGradKernel(const Context& ctx,
                            const DenseTensor& out_grad,
                            int downscale_factor,
                            const std::string& data_format,
                            DenseTensor* x_grad);
```

算子注册路径：

```c++
    paddle/phi/kernels/cpu/pixel_unshuffle_kernel.cc
    paddle/phi/kernels/cpu/pixel_unshuffle_grad_kernel.cc
    paddle/phi/kernels/gpu/pixel_unshuffle_kernel.cu
    paddle/phi/kernels/gpu/pixel_unshuffle_grad_kernel.cu
```
nn.Layer组网API实现路径: python/paddle/nn/layer/vision.py
函数API实现路径: python/paddle/nn/functional/vision.py
单元测试路径： python/paddle/fluid/tests/unittests/test_pixel_unshuffle.py

# 六、测试和验收的考量

考虑测试的情况：
- 计算结果跟Numpy基准值保持一致
- 前向和反向传播梯度计算的正确性；
- 支持NCHW以及NHWC输入数据格式， 不支持其他格式
- `downscale_factor`输入合法性校验, 是否是正整数，是否能整除。
- 覆盖CPU和GPU测试场景
- 覆盖静态图和动态图测试场景
- 入参的有效性和边界值测试

# 七、可行性分析和排期规划
已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面
`PixelUnshuffle`为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无


# paddle.nn.RReLU, paddle.nn.functional.rrelu设计文档

|API名称 | paddle.nn.RReLU, paddle.nn.functional.rrelu |
|---|---|
|提交作者 | thunder95 |
|提交时间 | 2022-03-29 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20200329_api_design_for_RReLU.md |


# 一、概述
## 1、相关背景
RRELU激活函数是从[Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853)中提出的，它是在Leaky ReLU基础上，对每一个位置的负值输入线性项做了随机采样 ，来加强一定范围内的泛化能力。 

## 2、功能目标

在飞桨框架中增加RReLU激活函数API。

## 3、意义

飞桨将支持RReLU激活函数API。

# 二、飞桨现状
飞桨目前不支持此功能， 需要基于飞桨框架新增OP。

计算原理相对简单，逻辑上可用下方的python代码模拟：

```python
import paddle
import numpy as np
import paddle.nn.functional as F
lower = 1 / 8.
upper = 1 / 3.
input = paddle.rand([2, 3, 4])
is_train = False #判断是否是训练状态
if is_train:
    alpha = paddle.uniform(input.shape, dtype='float32', min=lower, max=upper)
    alpha_x = alpha * input
    out = paddle.where(input >= 0, input, alpha_x)
else:
    negative_slope = (lower + upper) / 2.0
    out = F.leaky_relu(input, negative_slope)
print(out)
```

# 三、业内方案调研
## PyTorch

PyTorch中已经有了`torch.nn.RReLU`（文档在[这里](https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html?highlight=rrelu#torch.nn.RReLU)）。在Python层面的主要实现代码为：

```python
class RReLU(Module):
    __constants__ = ['lower', 'upper', 'inplace']

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self,
        lower: float = 1. / 8,
        upper: float = 1. / 3,
        inplace: bool = False
    ):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)
```

其中的`F.rrelu`是由C++实现的，主要代码为：

```c++
Tensor& rrelu_with_noise_out_cpu(const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator,
    Tensor& output) {
  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "rrelu_with_noise_out_cpu", [&] {
      _rrelu_with_noise_train<scalar_t>(output, self.contiguous(), noise, lower, upper, generator);
    });
    return output;
  } else {
    auto lower_tensor = scalar_to_tensor(lower);
    auto upper_tensor = scalar_to_tensor(upper);
    auto negative = (lower_tensor + upper_tensor) / 2;
    Scalar negative_slope = negative.item();
    return at::leaky_relu_out(output, self, negative_slope);
  }
}
```

TensorFlow目前也已经支持rrelu的addon:
```python
@tf.keras.utils.register_keras_serializable(package="Addons")
def rrelu(
    x: TensorLike,
    lower: Number = 0.125,
    upper: Number = 0.3333333333333333,
    training: Optional[bool] = None,
    seed: Optional[int] = None,
    rng: Optional[tf.random.Generator] = None,
) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    lower = tf.cast(lower, x.dtype)
    upper = tf.cast(upper, x.dtype)

    def random_a():
        if rng is not None and seed is not None:
            raise ValueError(
                "Either seed or rng should be specified. Not both at the same time."
            )

        if rng is not None:
            return rng.uniform(tf.shape(x), minval=lower, maxval=upper, dtype=x.dtype)

        return tf.random.uniform(
            tf.shape(x), minval=lower, maxval=upper, dtype=x.dtype, seed=seed
        )

    a = tf.keras.backend.in_train_phase(random_a, (lower + upper) / 2, training)

    return tf.where(x >= 0, x, a * x)
```

主要逻辑步骤包括:
1. 先确定是在训练或是测试状态, 二者需要分别处理
2. 测试状态下获取lower和upper的平均值, 在通过leakyRelu求得激活值
3. 训练状态下的negative slope是通过lower和upper之间的均匀采样获得

# 四、对比分析

使用Paddle的内置API进行组合迭代使用将会大大降低效率。
Pytorch支持inplace操作, 能有效降低显存占用。
计算逻辑上比较简单, 也可以在现在API的基础上进行拓展开发。


# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.nn.RRelu(lower=1. / 8., upper=1. / 3., name=None)`，

- **lower** (float) - 均匀分布的下边界。
- **upper** (float) - 均匀分布的上边界。

`paddle.nn.functional.rrelu(x, lower=1. / 8., upper=1. / 3., training=True, name=None)`,

- **x** (tensor) - 输入的张量。
- **lower** (float) - 均匀分布的下边界。
- **upper** (float) - 均匀分布的上边界。
- **training** (bool) - 是否在训练阶段, 默认是True。

## 底层OP设计
RRelu与飞桨中已有的PRelu在操作上是类似的，参考PReluOp设计来实现RRelu的底层OP。

## API实现方案

参考`paddle.nn.PRelu`来实现`paddle.nn.RRelu`，以及`paddle.nn.functional.rrelu`。

## 代码实现文件路径

CPU中正向和反向计算逻辑：
 paddle/phi/kernels/cpu/rrelu_grad_kernel.cc
 paddle/phi/kernels/cpu/rrelu_kernel.cc
 
GPU中正向和反向计算逻辑：
 paddle/phi/kernels/gpu/rrelu_funcs.h
 paddle/phi/kernels/gpu/rrelu_grad_kernel.cu
 paddle/phi/kernels/gpu/rrelu_kernel.cu
 
```c++
template <typename T, typename Context>
void RReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 DenseTensor* out,
                 DenseTensor* noise);
                          
template <typename T, typename Context>
void RReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& noise,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad);
```

算子注册路径：
paddle/fluid/operators/rrelu_op.cc

nn.Layer组网API实现路径: python/paddle/nn/layer/activation.py
函数API实现路径: python/paddle/nn/functional/activation.py
单元测试路径： python/paddle/fluid/tests/unittests/test_rrelu.py


# 六、测试和验收的考量

考虑测试的情况：
- 训练阶段无法固定与numpy相同随机值, 在测试阶段计算结果跟Numpy基准值保持一致
- 前向和反向传播梯度计算的正确性；
- `lower和upper`输入合法性校验, 是否在0和1之间的浮点数, upper应大于lower。
- 覆盖CPU和GPU测试场景
- 覆盖静态图和动态图测试场景
- 入参的有效性和边界值测试, 

# 七、可行性分析和排期规划
已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面
`RRelu`为独立新增API，对其他模块没有影响。

# 名词解释

无

# 附件及参考资料

无


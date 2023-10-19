#  LPPool1D / LPPool2D API 设计文档

| API 名称 | LPPool1D / LPPool2D |
| - | - |
| 提交作者 | WintersMontagne10335 |
| 提交时间 | 2023-10-19 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20231019_api_design_for_lp_pool.md |

# 一、概述

## 1、相关背景

平均池化是将输入的图像划分为若干个矩形区域，对每个子区域输出所有元素的平均值。平均池化可以更多保留图像的背景信息。

最大池化是将输入的矩阵划分为若干个矩形区域，对每个子区域输出最大值。最大池化可以更多保留图像的纹理信息。

幂平均池化是基于平均池化和最大池化的结合，利用一个学习参数p来确定这两种方法的相对重要性。当 p=1 时，等同于累加池化；当 p=∞ 时，等同于最大池化。

幂平均池化每个窗口的计算过程：

![39_1](https://github.com/WintersMontagne10335/doing/assets/118546135/eb928fe6-fff8-4c69-855a-82c933466583)

## 2、功能目标

新增 `LPPool1D / LPPool2D` API。

调用形式：
- `paddle.nn.LPPool1D`
- `paddle.nn.LPPool2D`
- `paddle.nn.functional.lp_pool1d`
- `paddle.nn.functional.lp_pool2d`

## 3、意义

为 `Paddle` 增加 `LPPool1D / LPPool2D` ，丰富 `Paddle` 中池化操作相关的 API。

# 二、飞桨现状

`Paddle` 目前已经提供了平均池化（ `AvgPool1d / AvgPool2d / AvgPool3d` ）、最大池化（ `MaxPool1d / MaxPool2d / MaxPool3d` ）等的池化方法。

目前 `Paddle` 在 `Python` 端缺少幂平均池化（ `LPPool1D / LPPool2D` ）相关接口的实现，而在底层也没有相关算子。

# 三、业内方案调研

## PyTorch

`Pytorch` 底层并未实现 `LPPool1D / LPPool2D` 对应的Kernel，而是通过在 `Python` 端，基于 `AvgPool1d / AvgPool2d`，组合实现了 API。

### API 文档

- [torch.nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)](https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#lppool1d)

    - sketch
        - Applies a 1D power-average pooling over an input signal composed of several input planes
        - Note
        > If the sum to the power of p is zero, the gradient of this function is not defined. This implementation will set the gradient to zero in this case

    - Parameters
        - kernel_size
        > a single int, the size of the window
        - stride
        > a single int, the stride of the window. Default value is kernel_size
        - ceil_mode
        >when True, will use ceil instead of floor to compute the output shape

- [torch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)](https://pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html#lppool2d)

    - sketch
        - Applies a 2D power-average pooling over an input signal composed of several input planes
        - Note
        > If the sum to the power of p is zero, the gradient of this function is not defined. This implementation will set the gradient to zero in this case

    - Parameters
        - kernel_size
        > the size of the window
        - stride
        > the stride of the window. Default value is kernel_size
        - ceil_mode
        > when True, will use ceil instead of floor to compute the output shape

### 实现逻辑 

因为 `LPPool1D` 与 `LPPool2D` 实现逻辑基本相同，所以以下仅对 `LPPool1D` 进行分析。

#### `Python` 端

关键源码

- [torch/nn/modules/pooling.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pooling.py#L891)

```Python
class _LPPoolNd(Module):
    __constants__ = ['norm_type', 'kernel_size', 'stride', 'ceil_mode']

    norm_type: float
    ceil_mode: bool

    def __init__(self, norm_type: float, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 ceil_mode: bool = False) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'norm_type={norm_type}, kernel_size={kernel_size}, stride={stride}, ' \
            'ceil_mode={ceil_mode}'.format(**self.__dict__)

class LPPool1d(_LPPoolNd):
    kernel_size: _size_1_t
    stride: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        return F.lp_pool1d(input, float(self.norm_type), self.kernel_size,
                           self.stride, self.ceil_mode)
```

调用了 `torch/nn/functional.py` 中的 `lp_pool1d` 函数。

- [torch/nn/functional.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L1050)

```Python
def lp_pool1d(
    input: Tensor, norm_type: Union[int, float],
    kernel_size: int,
    stride: Optional[BroadcastingList1[int]] = None,
    ceil_mode: bool = False
) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            lp_pool1d, (input,), input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode
        )
    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (torch.sign(out) * relu(torch.abs(out))).mul(kernel_size).pow(1.0 / norm_type)
```

`LPPool1d` 的公式如下:

![image](https://github.com/WintersMontagne10335/doing/assets/118546135/14e5652f-521c-4b2c-bda4-7857e481ec34)

`AVGPool1d` 的公式如下:
![image](https://github.com/WintersMontagne10335/doing/assets/118546135/9927757c-4c50-40aa-9e8a-ac3be01b54b0)

对比可知，`LPool1d` 开方符号下的部分，是变形的累加池化，而累加池化又可以通过平均池化的结果乘以 `kernel_size` 来实现。所以，`PyTorch` 基于 `AVGPool1d` ，并与 `mul` 等相组合，实现了 `LPool1d` 。
 
#### CPU端

`PyTorch` 未实现。

#### GPU端

`PyTorch` 未实现。

## TensorFlow

`TensorFlow` 未实现该算子。

## MXNet

`MXNet` 未实现该算子。

## OneFlow

`OneFlow` 未实现该算子。

# 四、对比分析

- 目前，主流深度学习框架仅有 `Pytorch` 实现了该算子。

- `Pytorch` 的 `LPPool1d` ,是通过在Python端，基于 `AVGPool1d` 组合实现的。 `Paddle` 的实现也可以沿用这个思路。这里存在一个小问题：[Paddle 的 AvgPool1d 与 PyTorch 的参数不一致](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.AvgPool1d.html)，但是并不影响本API的实现。

- 组合实现的方式，会对性能有较大的负面影响。针对该算子， `Paddle` 最终应该实现底层 Kernel 。

# 五、设计思路与实现方案

分两步来实现：

1. 基于 `AVGPool1d` 组合实现 API ，提交Python 实现代码、单测代码、中文API文档、英文API文档。
2. 实现底层 Kernel ，提交一个性能优化的 PR 。

`本设计文档仅涉及第一部分。不久会补充第二部分的设计文档。`
 
## 命名与参数设计

添加 python 上层接口:

- `paddle.nn.functional.lp_pool1d`

    ``` python
    paddle.nn.functional.lp_pool1d(
        input:Tensor,
        norm_type:Union[int, float],
        kernel_size:int,
        stride:Optional[BroadcastingList1[int]]=None,
        ceil_mode:bool=False)
    ```

    |参数名|类型|描述|
    |---|---|---|
    |input|Tensor|Tensor of input|
    |norm_type|Union[int, float]|exponent|
    |kernel_size|int|the size of the window|
    |stride|Optional[BroadcastingList1[int]]|the stride of the window|
    |ceil_mode|bool|when True, will use ceil instead of floor to compute the output shape|
    |output|Tensor|Tensor of output|

- `paddle.nn.LPPool1D`

   调用 `paddle.nn.functional.lp_pool1d` ，参数与之类似。

- `paddle.nn.functional.lp_pool2d`

    ``` python
    paddle.nn.functional.lp_pool2d(
        input:Tensor,
        norm_type:Union[int, float],
        kernel_size:BroadcastingList2[int],
        stride:Optional[BroadcastingList2[int]]=None,
        ceil_mode:bool=False)
    ```

    |参数名|类型|描述|
    |---|---|---|
    |input|Tensor|Tensor of input|
    |norm_type|Union[int, float]|exponent|
    |kernel_size|BroadcastingList2[int]|the size of the window|
    |stride|Optional[BroadcastingList2[int]]|the stride of the window|
    |ceil_mode|bool|when True, will use ceil instead of floor to compute the output shape|
    |output|Tensor|Tensor of output|

- `paddle.nn.LPPool2D`

   调用 `paddle.nn.functional.lp_pool2d` ，参数与之类似。

## 底层 OP 设计

本设计文档不涉及。

## API实现方案

- `paddle.nn.functional.lp_pool1d`

利用 `Paddle` 已有的 `paddle.nn.functional.avg_pool1d`组合实现。

- `paddle.nn.LPPool1D`

调用 `paddle.nn.functional.lp_pool1d` 实现。

- `paddle.nn.functional.lp_pool2d`

利用 `Paddle` 已有的 `paddle.nn.functional.avg_pool2d`组合实现。

- `paddle.nn.LPPool2D`

调用 `paddle.nn.functional.lp_poo21d` 实现。

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **输出正确性**
  输出数值结果的一致性和数据类型是否正确，使用 PyTorch 作为参考标准

- **计算精度**
  需要保证 `前向/后向` 计算的精度正确性，使用 PyTorch 作为参考标准

- **维度测试**
  需要测试 `1d / 2d` 两类接口

# 七、可行性分析及规划排期

最晚下周末完成第一阶段的工作。

# 八、影响面

新增 API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

- [torch.nn.functional.lp_pool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool1d.html)
- [torch.nn.LPPool1d](https://pytorch.org/docs/stable/generated/torch.nn.LPPool1d.html#lppool1d)
- [torch.nn.functional.lp_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool2d.html)
- [torch.nn.LPPool2d](https://pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html)

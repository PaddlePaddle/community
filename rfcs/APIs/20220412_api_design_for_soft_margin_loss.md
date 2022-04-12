#  paddle.nn.SoftMarginLoss 设计文档


|API名称 | SoftMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 
lvpengbo | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-04-12 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220412_api_design_for_SoftMarginLoss.md<br> | 


# 一、概述
## 1、相关背景
在 Paddle 框架中，需新增 SoftMarginLoss API，调用路径为：paddle.nn.SoftMarginLoss 和 paddle.nn.functional.soft_margin_loss
## 2、功能目标
paddle.nn.SoftMarginLoss 是一种计算二分类逻辑损失的函数
loss的公式为
$$loss(x,y)= \sum _i 
\frac{log(1+exp(−y[i]∗x[i]))}{x.nelement()}
)$$
## 3、意义
为 paddle 框架中新增计算损失函数的方法

# 二、飞桨现状

飞桨内已有margin_rank_loss,rank_loss,hinge_loss 等类似的应用于度量学习的计算loss的方法。

# 三、业内方案调研
Pytorch 中有相关的函数       
```
torch.nn.SoftMarginLoss(margin=1.0, 
                              p=2.0, 
                              eps=1e-06, 
                              swap=False, 
                              size_average=None, 
                              reduce=None, 
                              reduction='mean') -> Tensor
```

在 pytorch 中，介绍为：

> Creates a criterion that optimizes a two-class classification logistic loss between input tensor xx and target tensor yy (containing 1 or -1).
>$$loss(x,y)= \sum _i 
\frac{log(1+exp(−y[i]∗x[i]))}{x.nelement()}
)$$
>Shape:
Input: (*)(∗), where *∗ means any number of dimensions.    
> Target: (*)(∗), same shape as the input.    
>Output: scalar. If reduction is 'none', then (*)(∗), same shape as input.

PyTorch python 代码：
```
class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(SoftMarginLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.soft_margin_loss(input, target, reduction=self.reduction)

def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            soft_margin_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return torch._C._nn.soft_margin_loss(input, target, reduction_enum)
```

## Pytorch C++实现

PyTorch中SoftMarginLoss中cpu的实现位于pytorch/aten/src/ATen/native/Loss.cpp这个文件中

##### 前向传播
```cpp 
Tensor& soft_margin_loss_out(const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output) {
  // compute inplace variant of: output = at::log(1. + at::exp(-input * target));
  at::neg_out(output, input).mul_(target).exp_().add_(1.).log_();
  if (reduction != Reduction::None) {
    auto tmp = apply_loss_reduction(output, reduction);
    output.resize_({});
    output.copy_(tmp);
  }
  return output;
}
```

##### 反向传播

```cpp
Tensor& soft_margin_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto z = at::exp(-target * input);
  // inplace version of: grad_input = -norm * target * z / (1. + z) * grad_output;
  at::mul_out(grad_input, target, z).mul_(-norm);
  z.add_(1);
  grad_input.div_(z).mul_(grad_output);
  return grad_input;
}
```
Tensorflow 中暂未找到相关实现

# 四、对比分析
- 1.pytorch可以选择reduction方法,即"mean","sum","None"。 
- 看来pytorch的设计功能更加完善丰富一些，且pytorch框架与paddle相似，故采用pytorch的方案。。


# 五、设计思路与实现方案
为paddle phi计算库内部添加SoftMarginLoss的前向传播和反向传播大算子（CPU和GPU各自单独实现）。然后为paddle 动态图和静态图分别添加SoftMarginLoss的API。

## 命名与参数设计
共添加以下两个 API：
```
padde.nn.functional.soft_margin_loss(input Tensor[float64 or float32], target Tensor[float64 or float32],size_average [bool optional]默认为true,reduce [bool]作为参数，默认为true, reduction='mean', 'mean' 求平均,'sum'求和,'None',                                                negative, Tensor[float64 or float32],维度为[batch_size,dim]) -> Tensor
 
 
 
paddle.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean') -> Tensor
该函数底层调用padde.nn.functional.soft_margin_loss
```
## 底层OP设计
底层新增op算子
SoftMarginLossKernel(const DenseTensor& x,
                    const DenseTensor& y,
                    const std::string& reduction,
                    DenseTensor* out);
其中，x,y为输入，reduction为判断使用mean、sum、none的参数，out为输出。
## API实现方案
padde.nn.functional.soft_margin_loss API实现方案：
1. 检查参数

   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`target`）（同其余 functional loss 中的实现）
   3. 用reshape方法进行转换为维度[batch_size,dim],并检查参数维度是否相同。
 
2. 计算
   调用新增算子SoftMarginLossKernel，进行计算，返回结果

3. 输出 loss（同其余 functional loss 中的实现）

# 六、测试和验收的考量
测试考虑的case如下:

1.动态图，静态图，要与np计算下的结果输出需要一致。
2、1D，2D tensor的表现行为和pytorch表现一致
3.CPU、GPU下计算一致。
4.各reduction下计算一致
5.各参数输入有效。

# 七、可行性分析和排期规划
方案需新增Op算子，可以在当前版本周期内开发完成。

# 八、影响面
无影响

# 名词解释

# 附件及参考资料

op算子开发文档
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html 
pytorch相关文档
https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html
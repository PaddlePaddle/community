#  paddle.nn.MultiMarginLoss 设计文档


|API名称 | MultiMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-07 | 
|版本号 | v2.2| 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220707_api_design_for_multi_margin_loss.md<br> | 


# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，Paddle需要扩充 API paddle.nn.MultiMarginLoss 以及 paddle.functional.multi_margin_loss

## 2、功能目标
paddle.nn.MultiMarginLoss 为多分类问题的 Hinge loss。
loss 按照以下公式计算

$$\text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}$$

如果有权重 w 下，公式为

$$\text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}$$

## 3、意义
为 paddle 框架中新增计算损失函数的方法，完善并丰富飞桨深度学习框架的功能及实用性。

# 二、飞桨现状
Paddle 目前没有相关的损失函数方法。


# 三、业内方案调研
Pytorch 中有相关的
`torch.nn.functional.multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:`和`torch.nn.MultiMarginLoss(p: int = 1, 
                                         margin: float = 1.,
                                         weight: Optional[Tensor] = None, 
                                         size_average=None,
                                         reduce=None, 
                                         reduction: str = 'mean') -> None:`
其中的size_average和reduce将被弃用，统一使用reduction
在 pytorch 中，介绍为：
```
"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
    output :math:`y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)-1`):

    For each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar
    output :math:`y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D :attr:`weight` tensor into the constructor.

    The loss function then becomes:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}
"""
```

Pytorch 代码
```
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor
    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            multi_margin_loss,
            (input, target, weight),
            input,
            target,
            p=p,
            margin=margin,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")
    if weight is not None:
        if weight.dim() != 1:
            raise ValueError("weight must be one-dimensional")

    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)
```


tensorflow没有专门的multimarginloss

# 四、对比分析
- 采取使用pytorch的框架方法。


# 五、设计思路与实现方案

## 命名与参数设计
共添加以下两个 API：

- `paddle.nn.functional.multi_margin_loss(
    input,
    label,
    p:int=1,
    margin:float=1.0,
    weight= None,
    reduction: str = "mean",
    name:str=None,
) -> Tensor:`
    - input:Tensor, 维度为[batchsize,num_classes]
    - label:Tensor, 维度为[batchsize,]
    - weight: Optional[Tensor],维度为[num_classes,]
    - reduction:str,'none','mean','sum
    - name (str,可选)
和`
- paddle.nn.MultiMarginLoss(
    p:int = 1,
    margin:float=1.0,
    weight=None, 
    reduction='mean', 
    name=None,
) -> None:

## 底层OP设计
## API实现方案

1. 检查参数
    
   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`label`、`weight`）（同其余 functional loss 中的实现）
   3. 检查输入的`input`、`label`、`weight`维度是否正确。
   4. 检查 p， 只支持 1 和 2（同 pytorch 的实现）。

2. 计算

   1. 根据公式计算每一个 batchsize 的 loss。

3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）
# 六、测试和验收的考量

测试考虑的case如下:
- 1.动态图，静态图，要与numpy计算下的结果输出需要一致。
- 2.检查 p 的合法性。
- 3.检查 margin 的合法性
- 4.各reduction下计算一致
- 5.各参数输入有效。

 
# 七、可行性分析和排期规划
方案主要依赖现有paddle api组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面
无影响

#  paddle.nn.MultiLabelSoftMarginLoss 设计文档


|API名称 | MultiLabelSoftMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-19 | 
|版本号 | v2.0| 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220316_api_design_for_multilabel_soft_margin_loss.md<br> | 


# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，Paddle需要扩充APIpaddle.nn.MultiLabelSoftMarginLoss以及paddle.nn.functional.multilabel_soft_margin__loss
## 2、功能目标
paddle.nn.MultiLabelSoftMarginLoss 为多标签分类损失。
## 3、意义
为 paddle 框架中新增计算损失函数的方法

# 二、飞桨现状
MultiLabelSoftMarginLoss损失函数其实就是 sigmoid与BCEloss。
paddle都有这两个函数的实现，可以调用


# 三、业内方案调研
Pytorch 中有相关的

`torch.nn.functional.multilabel_soft_margin_loss(
    input: Tensor,
    label: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:`

和

`torch.nn.MultiLabelSoftMarginLoss(
    input: Tensor,
    label: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:`

其中的size_average和reduce将被弃用，统一使用reduction。

在 pytorch 中，介绍为：
```
"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input :math:`x` and target :math:`y` of size
    :math:`(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
                         + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\right\}`,
    :math:`y[i] \in \left\{0, \; 1\right\}`.
"""
```

Pytorch 代码
```
def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor
    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            multilabel_soft_margin_loss,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    class_dim = input.dim() - 1
    C = input.size(class_dim)
    loss = loss.sum(dim=class_dim) / C  # only return N loss values

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret
```


tensorflow没有专门的multilabelsoftmarginloss

# 四、对比分析
- BCELoss只使用ln函数计算，而multilabelsoftmarginloss使用sigmoid函数计算，采取使用pytorch的框架方法。


# 五、设计思路与实现方案

## 命名与参数设计
共添加以下两个 API：

`paddle.nn.functional.multilabel_soft_margin_loss(
    input,
    label,
    weight= None,
    reduction: str = "mean",
    name:str=None,
) -> Tensor:`
- input:Tensor, 维度为[batchsize,num_classes]
- label:Tensor, 维度为[batchsize,num_classes]
- weight: Optional[Tensor],维度为[batchsize,1]
- reduction: str, 'None' , 'mean' , 'sum'
- name (str, 可选)

和

`paddle.nn.MultiLabelSoftMarginLoss(
    weight,
    reduction,
    name,
) -> Tensor:`
- weight: Optional[Tensor], 维度为[batchsize,1]
- reduction: str, 'None' , 'mean' , 'sum'
- name: (str, 可选)

## 底层OP设计
## API实现方案
sigmoid函数可以通过 paddle.nn.functional.log_sigmoid()实现
1. 检查参数
    
   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`target`、`weight`）（同其余 functional loss 中的实现）
   3. 检查输入的`input`、`target`、`weight`维度是否相同

2. 计算

   1. 先计算loss
   2. 如果有权重weight，乘以权重
   3. 沿轴1将loss相加

3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）
# 六、测试和验收的考量

测试考虑的case如下:
- 1.动态图，静态图，要与np计算下的结果输出需要一致。
- 2.CPU、GPU下计算一致。
- 3.各reduction下计算一致
- 4.各参数输入有效。

 
# 七、可行性分析和排期规划
方案主要依赖现有paddle api组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面
无影响

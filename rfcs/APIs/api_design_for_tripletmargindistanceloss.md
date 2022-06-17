#  paddle.nn.TripletMarginWithDistanceLoss 设计文档


|API名称 | TripletMarginWithDistanceLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-16 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220316_api_design_for_tripletmargindistanceloss.md<br> | 


# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，Paddle需要扩充APIpaddle.nn.TripleMarginWithDistanceLoss以及paddle.nn.functional.triplet_margin_with_distance_loss
## 2、功能目标
paddle.nn.TripletMarginWithDistanceLoss 是三元损失函数，其针对样本和正负锚点计算任意给定距离函数下的三元损失，从而获得损失值。
## 3、意义
为 paddle 框架中新增计算损失函数的方法

# 二、飞桨现状
目前paddle缺少相关功能实现。
需要独立设计实现相关的函数。
飞桨内已有margin_rank_loss,rank_loss,hinge_loss 等类似的应用于度量学习的计算loss的方法。
可以在之前TripletMarginLoss的基础上设计。

# 三、业内方案调研
Pytorch 中有相关的
`torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`和`torch.nn.TripletMarginWithDistanceLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`

在 pytorch 中，介绍为：

> Creates a criterion that measures the triplet loss given an input tensors $x 1, x 2, x 3$ and a margin with a value greater than 0 . This is used for measuring a relative similarity between samples. A triplet is composed by a, $p$ and $n$ (i.e., anchor, positive examples and negative examples respectively). The shapes of all input tensors should be $(N, D)$.
>
> The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al.
> The loss function for each sample in the mini-batch is:
>
> $$
> L(a, p, n)=\max \left\{d\left(a_{i}, p_{i}\right)-d\left(a_{i}, n_{i}\right)+\operatorname{margin}, 0\right\}
> $$
>
Pytorch 代码
```
def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean"
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_with_distance_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )

    distance_function = distance_function if distance_function is not None else pairwise_distance

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output
```
- 定义distance_function，如果没有传入参数就设为2norm距离函数
- 分别计算样本与正负锚点的距离dist_pos,dist_neg，
- 如果swap为True，计算正负锚点的距离，将dist_neg改为 负锚点与样本间距离与正负锚点的距离之间 较小的值。
- 将dist_pos减去dist_neg加上margin，与0比较，取较大的值。
- reduction_enum 函数选择输出的方式包括 None、mean、sum 等

tensorflow 代码
```
def triplet_loss(queries, positives, negatives, margin=0.1):
  """Calculates Triplet Loss.
  Triplet loss tries to keep all queries closer to positives than to any
  negatives. Differently from the Contrastive Loss, Triplet Loss uses squared
  distances when computing the loss.
  Args:
    queries: [batch_size, dim] Anchor input tensor.
    positives: [batch_size, dim] Positive sample input tensor.
    negatives: [batch_size, num_neg, dim] Negative sample input tensor.
    margin: Float triplet loss loss margin.
  Returns:
    loss: Scalar tensor.
  """
  dim = tf.shape(queries)[1]
  # Number of `queries`.
  batch_size = tf.shape(queries)[0]
  # Number of `negatives`.
  num_neg = tf.shape(negatives)[1]

  # Preparing negatives.
  stacked_negatives = tf.reshape(negatives, [num_neg * batch_size, dim])

  # Preparing queries for further loss calculation.
  stacked_queries = tf.repeat(queries, num_neg, axis=0)

  # Preparing positives for further loss calculation.
  stacked_positives = tf.repeat(positives, num_neg, axis=0)

  # Computes *squared* distances.
  distance_positives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_positives), axis=1)
  distance_negatives = tf.reduce_sum(
      tf.square(stacked_queries - stacked_negatives), axis=1)
  # Final triplet loss calculation.
  loss = tf.reduce_sum(
      tf.maximum(distance_positives - distance_negatives + margin, 0.0))
  return loss
```


- 得到输入的batch_size和dim的大小，以及negatives的数目。
- 将queries，positives，negatives的维度都改为[num_neg*batch_size,dim]。
- 通过计算正负锚点与样本之间2范数距离，并分别求和得到 distance_positives 和 distance_negatives
- 通过 tf.maximum() ，计算出 loss

# 四、对比分析
- 1.pytorch对输入的数据维度进行一致性检测，并且支持任意distance_function包括lambda以及def的函数计算，tensorflow没有维度的检查，且只支持平方差计算。 
- 2.tensorflow没有swap和eps的参数选线，没有实现swap功能。 
- 3.pytorch可以选择reduction方法,即"mean","sum","None"。 
- 总体看来pytorch的设计功能更加完善丰富一些，且pytorch框架与paddle相似，故采用pytorch的方案。
# 五、设计思路与实现方案

## 命名与参数设计
共添加以下两个 API：

- `paddle.nn.TripletMarginWithDistanceLoss(distance_function=None, margin=1.0, swap=False, reduction='mean', name=None) -> Tensor`
- `padde.nn.functional.triplet_margin_with_distance_loss(input, positive, negative, distance_function=None, margin=1.0, swap=False, reduction='mean', name=None) -> Tensor`
## 底层OP设计
## API实现方案

1. 检查参数
    
   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`positive`、`negative`）（同其余 functional loss 中的实现）
   3. 检查参数维度是否相同。

2. 计算

   1. 用户可传入distance_function参数，如果未指定则使用 `paddle.nn.PairwiseDistance(2)` 分别计算得到正锚点与样本和负锚点与样本的距离。
   2. `swap` 参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
   3. 通过 `paddle.clip` 实现公式所示求出得 loss。

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

# 名词解释

# 附件及参考资料

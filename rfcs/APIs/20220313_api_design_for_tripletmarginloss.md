#  paddle.nn.TripletMarginLoss 设计文档


|API名称 | TripletMarginLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-13 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220313_api_design_for_tripletmarginloss.md<br> | 


# 一、概述
## 1、相关背景
Paddle需要扩充APIpaddle.nn.TripleMarginLoss以及paddle.nn.functional.triple_margin_loss
## 2、功能目标
paddle.nn.TripletMarginLoss 是三元损失函数，其针对 anchor 和正负对计算 P 范数距离下的三元损失，从而获得损失值。
## 3、意义
为 paddle 框架中新增计算损失函数的方法

# 二、飞桨现状

飞桨内已有margin_rank_loss,rank_loss,hinge_loss 等类似的应用于度量学习的计算loss的方法。

# 三、业内方案调研
Pytorch 中有相关的函数       
```
torch.nn.functional.triplet_margin_loss(anchor,
                                           positive, 
                                           negative, 
                                           margin=1.0, 
                                           p=2, 
                                           eps=1e-06, 
                                           swap=False, 
                                           size_average=None, 
                                           reduce=None, 
                                           reduction='mean') -> Tensor

torch.nn.TripletMarginLoss(margin=1.0, 
                              p=2.0, 
                              eps=1e-06, 
                              swap=False, 
                              size_average=None, 
                              reduce=None, 
                              reduction='mean') -> Tensor
```

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
> where
>
> $$
> d\left(x_{i}, y_{i}\right)=\left\|\mathbf{x}_{i}-\mathbf{y}_{i}\right\|_{p}
> $$

PyTorch C++ 代码：
```
Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  auto a_dim = anchor.dim();
  auto p_dim = positive.dim();
  auto n_dim = negative.dim();
  TORCH_CHECK(
      a_dim == p_dim && p_dim == n_dim,
      "All inputs should have same dimension but got ",
      a_dim,
      "D, ",
      p_dim,
      "D and ",
      n_dim,
      "D inputs.")
  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}
```
- 分别计算样本与正负锚点的距离dist_pos,dist_neg，
- 如果swap为True，计算正负锚点的距离，将dist_neg改为 负锚点与样本间距离与正负锚点的距离之间 较小的值。
- 将dist_pos减去dist_neg加上margin，与0比较，取较大的值。
- apply_loss_redution() 函数选择输出的方式包括（` mean`、`sum` 等）

Tensorflow python 代码
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
 整体逻辑为：

- 得到输入的batch_size和dim的大小，以及negatives的数目。
- 将queries，positives，negatives的维度都改为[num_neg*batch_size,dim]。
- 通过计算正负锚点与样本之间2范数距离，并分别求和得到 distance_positives 和 distance_negatives
- 通过 `tf.maximum()` ，计算出 loss

# 四、对比分析
- 1.pytorch对输入的数据维度进行一致性检测，并且支持p范数的计算，tensorflow没有维度的检查，只支持平方差计算。
- 2.tensorflow没有swap和eps的参数，没有实现swap功能。 
- 3.pytorch可以选择reduction方法,即"mean","sum","None"。 
- 总体看来pytorch的设计功能更加完善丰富一些，且pytorch框架与paddle相似，故采用pytorch的方案。。


# 五、设计思路与实现方案

## 命名与参数设计
共添加以下两个 API：

```
padde.nn.functional.triplet_margin_loss(input Tensor[float64 or float32] 维度为[batch_size,dim] 
                                          positive, Tensor[float64 or float32],维度为[batch_size,dim]                                                                                                 negative, Tensor[float64 or float32],维度为[batch_size,dim] 
                                          margin=1.0,
                                          p=2.0, 求距离时的范数,
                                          epsilon=1e-06,误差参数swap=False, 
                                          reduction='mean', 'mean' 求平均,'sum'求和,'None'直接输出维度为[batch_size,1]
                                          name=None) -> Tensor
 paddle.nn.TripletMarginLoss(margin=1.0  
                              p=2.0  
                              epsilon=1e-06                              
                              swap=False, 
                              reduction='mean', 
                              name=None) -> Tensor
```
## 底层OP设计
## API实现方案
1. 检查参数

   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`positive`、`negative`）（同其余 functional loss 中的实现）
   3. 用reshape方法进行转换为维度,[batch_size,dim],并检查参数维度是否相同。
 
2. 计算

   1. 使用 `paddle.linalg.norm` 分别计算得到正锚点与样本和负锚点与样本的距离。
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

# paddle.nn.TripletMarginLoss 设计文档

| API 名称     | paddle.nn.TripletMarginLoss                 |
| ------------ | ------------------------------------------- |
| 提交作者     | Ainavo                                      |
| 提交时间     | 2022-03-11                                  |
| 版本号       | V1.0                                        |
| 依赖飞桨版本 | v2.2.2                                      |
| 文件名       | 20220312_design_for_triplet_margin_loss.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算领域 API，Paddle 需要扩充 API `paddle.nn.TripletMarginLoss` 。

## 2、功能目标

增加 API `paddle.nn.TripletMarginLoss` ，实现三元损失函数。

## 3、意义

丰富 paddle 中的 loss 库，增加三元损失函数 API。

# 二、飞桨现状

目前 paddle 缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch 中有 functional API `torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`，以及对应的 Module `torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`

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

### 实现方法

在实现方法上，Pytorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/701fa16eed40c633d8eef6b4f04ab73a75c24749/aten/src/ATen/native/Loss.cpp?q=triplet_margin_loss#L148)。
C++ 代码实现如下：

```c++
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

整体逻辑为：

- 检查输入 `anchor`、`positive`、`negative` 三者的维度是否相等，不等报错
- 通过 `pairwise_distance()` 函数，分别计算 `anchor` 和 `positive` 之间的距离，以及 `anchor` 和 `negative` 之间的距离。
- `swap` 参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
- 通过 `clamp_distance()` 实现核心公式，计算出 `loss`。
- `apply_loss_redution()` 函数选择输出的方式包括（` mean`、`sum` 等）

## TensorFlow

### 实现方法

在实现方法上 tensorflow 以 python API 组合实现，[代码位置](https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/delf/delf/python/training/losses/ranking_losses.py)。

其中核心代码为：

```Python
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

- 读取输入`queries` 的维度、`batch_size` 大小。
- 读取输入 `negatives` 的数量并 `reshape()` 成 `[num_neg * batch_size, dim]` 的形状，对 `positives` 进行相同操作。
- 通过 `tf.square()` 函数计算欧式距离，`tf.reduce_sum()` 函数沿第二根轴求和，分别得到 `distance_positives` 和 `distance_negatives`
- 通过 `tf.maximum()` 实现核心公式，计算出 loss

# 四、对比分析

- 使用场景与功能：Pytorch 实现求解三元组 API 的基本功能，TensorFlow 以训练中的实际参数为代入，两种代码风格不同。功能上基本一致，这里 paddle 三元组 API 的设计将对齐 Pytorch 中的三元组 API。

# 五、方案设计

## 命名与参数设计

共添加以下三个 API：

- `paddle.nn.TripletMarginLoss(margin=1.0, p=2.0, epsilon=1e-06, swap=False, reduction='mean', name=None) -> Tensor`
- `padde.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, epsilon=1e-06, swap=False, reduction='mean', name=None) -> Tensor`
- `padde.nn.functional.pairwise_distance(x, y, p=2.0, epsilon=1e-06, keepdim=False, name=None) -> Tensor` （后续详述为何要添加此 API）

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

### nn.functional.pairwise_distance

该 API 实现于 `Paddle\python\paddle\nn\functional\distance.py`（目前尚无该文件，故需要新建）

由于 `nn.functional.triplet_margin_loss` 的实现过程中需要 `pairwise_distance`，但 paddle 目前没有 `pairwise_distance` 这样的 functional API，只有 `nn.PairwiseDistance` 这一 Layer API，不方便复用，因此先将 `nn.PairwiseDistance` API 的计算逻辑提取到 `nn.functional.pairwise_distance` 并暴露（已经调研过 torch 也有 `torch.nn.functional.pairwise_distance` 这样的 functional API）

实现逻辑同现有的 `nn.PairwiseDistance`，只不过按照 functional 的风格来写。

### nn.functional.triplet_margin_loss

该 API 实现于 `Paddle\python\paddle\nn\functional\loss.py`，与 `binary_cross_entropy`、`binary_cross_entropy_with_logits` 等函数放在一起。

1. 检查参数

   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `anchor`、`positive`、`negative`）（同其余 functional loss 中的实现）
   3. 检查 `anchor`、`positive`、`negative` 三者维度是否对齐

2. 计算

   1. 使用 `nn.functional.pairwise_distance` 分别计算得到正锚点与样本和负锚点与样本的距离。
   2. `swap` 参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
   3. 通过 `paddle.clip` 实现公式所示求出得 loss。

3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）

### nn.TripletMarginLoss

该 API 实现于 `Paddle\python\paddle\nn\layer\loss.py`，与 `BCEWithLogitsLoss`、`CrossEntropyLoss` 等类放在一起。

实现逻辑为调用 functional API `nn.functional.triplet_margin_loss`，与其他 loss Layer 保持一致。

# 六、测试和验收的考量

测试考虑的 case 如下：

- `padde.nn.functional.triplet_margin_loss`, `paddle.nn.TripletMarginLoss` 和 torch 结果是否一致；
- 参数 `margin` 为 float 和 1-D Tensor 时输出的正确性；
- 参数 `p` 各个取值的正确性；
- 参数 `epsilon` 的正确性；
- 参数 `swap` 为 `True` 或者 `False` 的正确性；
- 输入含 `NaN` 结果的正确性；
- `reduction` 对应不同参数的正确性；
- 错误检查：`p` 值 `p<1` 时能正确抛出错误

# 七、可行性分析及规划排期

方案主要依赖现有 paddle api 组合而成，且依赖的 `paddle.clip`、`paddle.min` 已于前期合入，依赖的 `paddle.nn.functional.pairwise_distance` 从 `paddle.nn.PairwiseDistance` 提取得到。

具体规划为

- 阶段一：提取 `nn.PairwiseDistance` 主要逻辑到 `nn.functional.pairwise_distance`，在 `nn.PairwiseDistance` 中调用它，保证其逻辑不变
- 阶段二：完成 `nn.functioanl.triplet_margin_loss`，并在 `nn.TripletMarginLoss` 中调用
- 阶段三：完成 `nn.functioanl.triplet_margin_loss` 单元测试
- 阶段四：为三个新的 API 书写中文文档

# 八、影响面

除去本次要新增的两个 API，额外增加了一个 `nn.functional.pairwise_distance`，但对原有的 `nn.PairwiseDistance` 没有影响

# 名词解释

无

# 附件及参考资料

无

#  paddle.nn.TripletMarginDistanceLoss 设计文档


|API名称 | TripletMarginDistanceLoss | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | yangguohao | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-16 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220316_api_design_for_tripletmargindistanceloss.md<br> | 


# 一、概述
## 1、相关背景
为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充APIpaddle.nn.TripleMarginDistanceLoss以及paddle.nn.functional.triplet_margin_with_distance_loss
## 2、功能目标
paddle.nn.TripletMarginDistanceLoss 是三元损失函数，其针对 anchor 和正负对计算 P 范数距离下的三元损失，从而获得损失值。
## 3、意义
为 paddle 框架中新增计算损失函数的方法

# 二、飞桨现状
对飞桨框架目前支持此功能的现状调研，如果不支持此功能，如是否可以有替代实现的API，是否有其他可绕过的方式，或者用其他API组合实现的方式；
目前paddle缺少相关功能实现。
需要独立设计实现相关的函数

# 三、业内方案调研
Pytorch 中有相关的`torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`和`torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean') -> Tensor`

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


# 四、对比分析
paddle 和 pytorch整体框架相似，故直接采用其方法进行设计。

# 五、设计思路与实现方案

## 命名与参数设计
参考：[飞桨API 设计及命名规范]
共添加以下两个 API：

- `paddle.nn.TripletMarginDistanceLoss(margin=1.0, distance_function=None, swap=False, reduction='mean', name=None) -> Tensor`
- `padde.nn.functional.triplet_margin_with_distance_loss(input, positive, negative, distance_function=None, margin=1.0, swap=False, reduction='mean', name=None) -> Tensor`
## 底层OP设计
## API实现方案
distance functions可以采用paddle.nn.PairWiseDistance来进行实现
1. 检查参数

   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 dtype（含 `input`、`positive`、`negative`）（同其余 functional loss 中的实现）

2. 计算

   1. 用户可传入distance_function参数，如果未指定则使用 `paddle.nn.PairWiseDistance` 分别计算得到正锚点与样本和负锚点与样本的距离。
   2. `swap` 参数判断：正锚点和负锚点间距离，并与负锚点与样本间距离进行比较，取更小的距离作为负锚点与样本间的距离。
   3. 通过 `paddle.clip` 实现公式所示求出得 loss。

3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）
# 六、测试和验收的考量

测试考虑的case如下:
1.动态图，静态图，要与np计算下的结果输出需要一致。
2.自定义distanc_function动态图静态图下输出一致。
2.在swap下，动态图静态图输出结果一致。

# 七、可行性分析和排期规划
方案主要依赖现有paddle api组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面
无影响

# 名词解释

# 附件及参考资料

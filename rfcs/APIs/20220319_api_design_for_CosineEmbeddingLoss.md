# 为 Paddle 新增 paddle.nn.CosineEmbeddingLoss 和 paddle.nn.functional.cosine_embedding_loss设计文档

| API名称      | paddle.optimizer.lr.CyclicLR               |
| ------------ | ------------------------------------------ |
| 提交作者     | NetPunk                                    |
| 提交时间     | 2022-03-19                                 |
| 版本号       | V1.0                                       |
| 依赖飞桨版本 | v2.2.2                                     |
| 文件名       | 20220318_design_for_CosineEmbeddingLoss.md |

# 一、概述

## 1、相关背景

`CosineEmbeddingLoss`即余弦相似度损失函数，用于判断输入的两个向量是否相似。 常用于非线性词向量学习以及半监督学习。

## 2、功能目标

在 Paddle 框架中，增加两个API：CosineEmbeddingLoss和cosine_embedding_loss，调用路径为：`paddle.nn.CosineEmbeddingLoss` 和 `paddle.nn.functional.cosine_embedding_loss`

## 3、意义

飞桨支持余弦相似度损失函数

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

pytorch中由API `torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')`以及对应的`torch.nn.functional.cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')->Tensor`，在pytorch中，介绍为：

~~~
Creates a criterion that measures the loss given input tensors x1, x2 and a Tensor label y with values 1 or -1. This is used for measuring whether two inputs are similar or dissimilar, using the cosine distance, and is typically used for learning nonlinear embeddings or semi-supervised learning.
~~~

### 实现方法

在实现方法上, Pytorch是通过C++ API组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/1522912602bc4cc5f7adbce66cad00ebb436f195/aten/src/ATen/native/Loss.cpp#L100)。

核心代码为：

```c++
Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  auto targ_dim = target.dim();
  TORCH_CHECK(
      targ_dim == 1 || targ_dim == 0,
      "0D or 1D target tensor expected, multi-target not supported");

  if (targ_dim == 1) {
    TORCH_CHECK(
        input1.dim() == 2,
        "1D target tensor expects 2D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        input1.dim() == 1,
        "0D target tensor expects 1D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");
  }

  auto prod_sum = (input1 * input2).sum(targ_dim);
  auto mag_square1 = (input1 * input1).sum(targ_dim) + EPSILON;
  auto mag_square2 = (input2 * input2).sum(targ_dim) + EPSILON;
  auto denom = (mag_square1 * mag_square2).sqrt_();
  auto cos = prod_sum / denom;

  auto zeros = at::zeros_like(cos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto pos = 1 - cos;
  auto neg = (cos - margin).clamp_min_(0);
  auto output_pos = at::where(target == 1, pos, zeros);
  auto output_neg = at::where(target == -1, neg, zeros);
  auto output = output_pos + output_neg;
  return apply_loss_reduction(output, reduction);
}
```

上面算法的大致流程和实际的数学公式计算类似：

1. 首先判定输入维度是否正确
2. 计算向量点乘`prod_sum`和向量二阶范式乘积`denom`
3. 求出余弦相似度`cos = prod_sum / denom`
4. 对正负样本结果做不同处理，即`output = output_pos + output_neg`
   * 当`target`为1时，`output_pos`为`1 - cos`，此时`output_neg`为0
   * 当`target`为1时，且如果`cos - margin`大于0，`output_neg`为`cos - margin`，否则为0，此时`output_pos`为0
5. 利用已实现API进行结果后处理`apply_loss_reduction(output, reduction)`

## 其他实现

tensorflow没有实现CosineEmbeddingLoss，但是实现了余弦相似度的计算

介绍为：

```
Computes the cosine similarity between labels and predictions.
```

### 实现方法

直接使用tensorflow库中以实现的API进行组合，[代码位置](https://github.com/tensorflow/tensorflow/blob/e43633feb66a7a239875aa86f3e5f479488ee570/tensorflow/python/keras/losses.py#L1950)

核心代码为：

```python
def cosine_similarity(y_true, y_pred, axis=-1):
 	y_true = nn.l2_normalize(y_true, axis=axis)
  	y_pred = nn.l2_normalize(y_pred, axis=axis)
  	return -math_ops.reduce_sum(y_true * y_pred, axis=axis)
```

# 四、对比分析

只在pytorch中找到了完整实现，因此不进行对比分析。

# 五、方案设计

## 命名与参数设计

CosineEmbeddingLoss的API设计为`paddle.nn.CosineEmbeddingLoss(margin=0, reduction='mean', name=None)`，cosine_embedding_loss的API设计为`paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0, reduction='mean', name=None)`，其中：

* margin：余弦相似度损失函数中的margin值
* reduction：结果后处理的类型，可以为`mean`或者`sum`
* input1和input2：输入的两个tensor
* label：真实的类别标签
* name：操作的名称，更多信息请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)。

在pytorch中，CosineEmbeddingLoss还有`size_average`、`reduce`两个参数，但是已经弃用，其功能转移到`reduction`参数上。两个参数的描述文档如下

~~~
size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True

reduce (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
~~~

通过查看pytorch源码可以确认，两个参数只是引用时可以传入，但是之后不再调用

~~~python
    def __init__(self, margin: float = 0., size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CosineEmbeddingLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)
~~~

在cosine_embedding_loss中，也有`size_average`、`reduce`两个参数，并且默认为`None`，下面是cosine_embedding_loss部分源码

~~~python
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
~~~

可以看到只有当`size_average`或`reduce`不为`None`时，才会作为`reduction`的替代，并由C++算子调用

因此此处省略`size_average`、`reduce`两个参数

## 底层OP设计

直接使用paddle API组合实现，不再单独设计OP。

## API实现方案

主要参考Pytorch进行实现，CosineEmbeddingLoss实现位置为`paddle/nn/layer/loss.py`，cosine_embedding_loss实现位置为`paddle/nn/functional/loss.py`

1. 使用`paddle.zero`初始化结果列表
2. 使用`paddle.matmul`实现向量点乘
3. 使用`paddle.norm`实现向量二次范数相乘
4. 利用python语句和内置`max`函数实现正负样本分开处理
   * 若标签为1，正样本损失值为`1 - 余弦相似度`
   * 若标签为-1，`余弦相似度-margin`小于0，负样本损失值为0
   * 若标签为-1，`余弦相似度-margin`大于0，负样本损失值为`余弦相似度-margin`
5. 使用paddle API`sum` 和`mean` 实现reduction计算

# 六、测试和验收的考量

测试考虑的case如下：

- 动态图，静态图，与numpy的结果保持一致；
- 输入含`NaN`结果的正确性；
- 在`cpu`和`gpu`环境下结果保持一致
- 支持`float64`、`float32`、`int64`、`int32`类型变量输入;
- 错误检查：`input`和 `target`维度不合规时能抛出输入维度错误；
- 错误检查：`margin`设置超出[-1, 1]范围时抛出参数设置错误；
- 错误检查：`reduction`设置除`sum` 和`mean`以外时抛出参数设置错误；

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无


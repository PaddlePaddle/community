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

pytorch中由API `torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')`以及对应的`torch.nn.functional.``cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')->Tensor`，在pytorch中，介绍为：

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
4. 计算出正负样本损失值总和`output`
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

API设计为`paddle.optimizer.lr.CyclicLR(base_learning_rate,max_learning_rate,step_size_up,step_size_down, mode='triangular',gamma=1.,scale_fn=None,scale_mode='cycle',last_epoch=-1,verbose=False)`

去除了Pytorch中`momentum`的相关参数。

同时，为了保持与paddle其他lrscheduler相关的api保持一致，将`base_lr`修改为`base_learning_rate`，`max_lr`修改为`max_learning_rate`。

## 底层OP设计

直接使用paddle API组合实现，不再单独设计OP。

## API实现方案

主要参考Pytorch进行实现，CosineEmbeddingLoss实现位置为`paddle/nn/layer/loss.py`，cosine_embedding_loss实现位置为`paddle/nn/functional/loss.py`

1. 使用`paddle.zero`初始化结果列表
2. 使用`paddle.matmul`实现向量点乘
3. 使用`paddle.norm`实现向量二次范数相乘
4. 使用paddle API`sum` 和`mean` 实现reduction计算

# 六、测试和验收的考量

测试考虑的case如下：

- 动态图，静态图，与numpy的结果保持一致；
- 输入含`NaN`结果的正确性；
- 错误检查：`input`和 `target`维度不为1-0或2-1时能抛出输入维度错误；

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无


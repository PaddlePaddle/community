# paddle.nn.functional.pairwise_distance 设计文档

| API 名称     | paddle.nn.functional.pairwise_distance   |
| ------------ | ---------------------------------------- |
| 提交作者     | Ainavo                                   |
| 提交时间     | 2022-07-06                               |
| 版本号       | V1.0.0                                   |
| 依赖飞桨版本 | develop                                  |
| 文件名       | 20220706_design_for_pairwise_distance.md |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持神经网络搭建相关 API，Paddle 需要扩充 API `paddle.nn.functional.pairwise_distance`。

## 2、功能目标

增加 API `paddle.nn.functional.pairwise_distance`，用于计算两组向量两两之间的距离。计算方式如下：
**p-norm(x - y + epsilon, p, last_dim, keepdim)**
p-norm 计算函数如下：

> $$
> \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p} p \geq 1, x \in R ^ n
> $$

- 对不同 shape 的输入，当 x 和 y 分别取 (N, D) 和 (D, )、(N, D) 和 (N, D)、(D, ) 和 (N, D) 以及 (D, ) 和 (D, ) 这四种情况，都能正确的广播并计算相应的范数；
- 对于不同的 p 值，可以求解不同类型的 p 范数；
- 参数 `keepdim` 可以控制输出的维度是否与输入保持一致。

## 3、意义

为 Paddle 增加神经网络相关的距离计算函数，丰富 `paddle.nn` 中的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 functional API `paddle.nn.functional.pairwise_distance`，但是存在 class API `paddle.nn.PairwiseDistance(p=2., epsilon=1e-6, keepdim=False, name=None)`，参考 Paddle 其他的 `layer` 和 `functional` 下的文件是一一对应的关系，因此在需要在 `functional` 目录下添加该 API。
- 该 API 的实现及测试，主要参考目前 Paddle 中含有的 `paddle.linalg.norm` 和 `paddle.dist` API，下面是对这两个 API 的补充说明。

## [paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/norm_cn.html#norm)

计算给定 Tensor 的矩阵范数（Frobenius 范数）和向量范数（向量 1 范数、 2 范数、或者通常的 p 范数）

- `x`(`Tensor`) ：维度为多维，数据类型为 `float32` 或 `float64`，
- `p` (`float|string`，可选) ： 范数(ord)的种类。目前支持的值为 `fro`、`inf`、`-inf`、`0`、`1`、`2`，和任何正实数 p 对应的 p 范数。默认值为 `fro`。
- `axis` (`int|list|tuple`，可选) ：使用范数计算的轴。如果 axis 为`None` ，则忽略 `input` 的维度，将其当做向量来计算。如果 axis 为 `int` 或者只有一个元素的 `list|tuple` ，norm API 会计算输入 `Tensor` 的向量范数。如果 axis 为包含两个元素的 list ，API 会计算输入 `Tensor` 的矩阵范数。 当 `axis < 0` 时，实际的计算维度为 `rank(input) + axis`。默认值为 `None`。
- `keepdim` (`bool`，可选) ：是否在输出的 Tensor 中保留和输入一样的维度，默认值为 False。当 keepdim 为 False 时，输出的 Tensor 会比输入 `input` 的维度少一些。
- `name` (`str|None`) ：该参数供开发人员打印调试信息时使用。默认值为`None`。

## [paddle.dist(x, y, p=2)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist)

该 API 用于计算 (x-y) 的 p 范数（p-norm），需要注意这不是严格意义上的范数，仅作为距离的度量。输入 x 和 y 的形状（shape）必须是可广播的（broadcastable）。

- `x`(`Tensor`)：1-D 到 6-D Tensor，数据类型为 `float32` 或 `float64`。
- `y`(`Tensor`)：1-D 到 6-D Tensor，数据类型为 `float32` 或 `float64`。
- `p`(`float，optional`)：用于设置需要计算的范数，数据类型为 `float32` 或 `float64`。默认值为 2。

## 两者的异同点

相同点：

- 两个 API 皆是计算 p 范数。
- 输入都可以是高阶张量。
- 对于输入的 `x` 和 `y` 都支持广播（broadcastable）。

不同点：

- paddle.linalg.norm 的`p`可以取值 `fro`（Frobenius 范数），paddle.dist 的 `p` 默认为 2。
- paddle.linalg.norm 相比于 paddle.dist 增加了可选参数 `axis`：使用范数计算的轴以及 `keepdim` 输出 `Tensor` 时是否保持维度与输入一致，增加了 API 在不同情况下使用的灵活性。

# 三、业内方案调研

## PyTorch

PyTorch 中有 functional API `torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False) -> Tensor`，以及对应的 class API `torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)`。

在 PyTorch 中，介绍为：

> Computes the pairwise distance between vectors :math: $v_1$, :math: $v_2$ using the p-norm:

> $$
> \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.
> $$

### 实现方法

在实现方法上，PyTorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/9e137ee583c4fdb2dd3aa0c425dc9c289454cbf2/aten/src/ATen/native/Distance.cpp)。
C++ 代码实现如下：

```c++
// pytorch/aten/src/ATen/native/Distance.cpp
Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  // Since either x1 or x2 could be broadcasted
  auto x1_dim = x1.dim();
  auto x2_dim = x2.dim();
  auto output_dim = x1_dim > x2_dim ? x1_dim : x2_dim;
  auto innermost_dim = output_dim - 1;
  return at::norm(x1 - x2 + eps, p, innermost_dim, keepdim);
}

// pytorch/aten/src/ATen/native/ReduceOps.cpp
TORCH_IMPL_FUNC(norm_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, c10::nullopt, result);
}

TORCH_IMPL_FUNC(norm_dtype_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, dtype, result);
}

void impl_func_norm(
    const Tensor& self,
    const OptionalScalarRef& opt_p,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    const Tensor& result) {
  auto p = opt_p.has_value() ? opt_p.get() : Scalar(2.0).to<double>();
  auto in_dtype = opt_dtype.value_or(self.scalar_type());
  auto out_dtype = result.scalar_type();

  // See the note [Reductions do not use vectorized ops]
  Tensor self_;
  if (self.is_cpu() && self.is_complex() && std::abs(p.toDouble()) == INFINITY) {
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype).abs();
    } else {
      self_ = self.abs();
    }
  } else {
    self_ = self;
  }


  // omit in_dtype in the following call, to avoid make_reduction explicitly
  // casting input to out_dtype
  auto iter = isComplexType(self_.scalar_type())
      ? meta::make_reduction(self_, result, dim, keepdim, in_dtype)
      : meta::make_reduction_from_out_ty(self_, result, dim, keepdim, out_dtype);

  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter.device_type(), iter, p);
  }
}

```

整体逻辑为：

- 获取 x1、x2 的维度，输出维度为输入中维度大的向量，以实现广播的效果。
- innermost_dim：输入维度-1，即最后一维（轴）
- 调用 ATen 张量库中的 norm 函数。

## TensorFlow

TensorFlow 中有 class API `nsl.keras.layers.PairwiseDistance(distance_config=None, **kwargs)`以及距离函数 functional API `nsl.lib.pairwise_distance_wrapper(sources, targets, weights=1.0, distance_config=None)`

### 实现方法

在实现方法上 TensorFlow 以 Python API 组合实现，[代码位置](https://github.com/tensorflow/neural-structured-learning/blob/c21dad4feff187cdec041a564193ea7b619b8906/neural_structured_learning/lib/distances.py#L222)。

Python 代码实现如下：

```python
def pairwise_distance_wrapper(sources,
                              targets,
                              weights=1.0,
                              distance_config=None):
  """A wrapper to compute the pairwise distance between `sources` and `targets`.
  `distances = weights * distance_config.distance_type(sources, targets)`
  This wrapper calculates the weighted distance between `(sources, targets)`
  pairs, and provides an option to return the distance as the sum over the
  difference along the given axis, when vector based distance is needed.
  For the usage of `weights` and `reduction`, please refer to `tf.losses`. For
  the usage of `sum_over_axis`, see the following examples:
  Given target tensors with shape `[batch_size, features]`, the reduction set to
  `tf.compat.v1.losses.Reduction.MEAN`, and `sum_over_axis` set to the last
  dimension, the weighted average distance of sample pairs will be returned.
  For example: With a distance_config('L2', sum_over_axis=-1), the distance
  between [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]]
  will be {(0+0) + (4+0) + (16+4) + (16+1)}/4 = 10.25
  If `sum_over_axis` is `None`, the weighted average distance of feature pairs
  (instead of sample pairs) will be returned. For example: With a
  distance_config('L2'), the distance between
  [[1, 1], [2, 2], [0, 2], [5, 5]] and [[1, 1], [0, 2], [4, 4], [1, 4]] will be
  {(0+0) + (4+0) + (16+4) + (16+1)}/8 = 5.125
  If `transform_fn` is not `None`, the transform function is applied to both
  `sources` and `targets` before computing the distance. For example:
  `distance_config('KL_DIVERGENCE', sum_over_axis=-1, transform_fn='SOFTMAX')`
  treats `sources` and `targets` as logits, and computes the KL-divergence
  between the two probability distributions.
  Args:
    sources: `Tensor` of type `float32` or `float64`.
    targets: `Tensor` of the same type and shape as `sources`.
    weights: (optional) `Tensor` whose rank is either 0, or the same as that of
      `targets`, and must be broadcastable to `targets` (i.e., all dimensions
      must be either `1`, or the same as the corresponding distance dimension).
    distance_config: An instance of `nsl.configs.DistanceConfig` that contains
      the following configuration (or hyperparameters) for computing distances:
      (a) `distance_type`: Type of distance function to apply.
      (b) `reduction`: Type of distance reduction. See `tf.losses.Reduction`.
      (c) `sum_over_axis`: (optional) The distance is the sum over the
        difference along the specified axis. Note that if `sum_over_axis` is not
        `None` and the rank of `weights` is non-zero, then the size of `weights`
        along `sum_over_axis` must be 1.
      (d) `transform_fn`: (optional) If set, both `sources` and `targets` will
        be transformed before calculating the distance. If set to 'SOFTMAX', it
        will be performed on the axis specified by 'sum_over_axis', or -1 if the
        axis is not specified. If `None`, the default distance config will be
        used.
  Returns:
    Weighted distance scalar `Tensor`. If `reduction` is
      `tf.compat.v1.losses.Reduction.MEAN`, this has the same shape as
      `targets`.
  Raises:
    ValueError: If the shape of targets doesn't match that of sources, or if the
      shape of weights is invalid.
    TypeError: If the distance function gets an unexpected keyword argument.
  """
  if distance_config is None:
    distance_config = configs.DistanceConfig()  # Default configs.

  tf.compat.v1.losses.Reduction.validate(distance_config.reduction)

  if distance_config.transform_fn is not configs.TransformType.NONE:
    sources = _apply_transform(sources, distance_config.transform_fn,
                               distance_config.sum_over_axis)
    targets = _apply_transform(targets, distance_config.transform_fn,
                               distance_config.sum_over_axis)

  sum_over_axis = distance_config.sum_over_axis
  # Validates the `sum_over_axis`
  _assert_valid_axis(sources.get_shape().ndims, sum_over_axis)
  distance_fn = _select_distance_fn(distance_config.distance_type)
  if distance_config.distance_type == configs.DistanceType.COSINE:
    # Cosine distance function assumes input tensors have been unit-normalized
    sources = tf.nn.l2_normalize(sources, axis=sum_over_axis)
    targets = tf.nn.l2_normalize(targets, axis=sum_over_axis)
  if _is_axis_required_in_distance_fn(distance_config.distance_type):
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        axis=sum_over_axis,
        reduction=distance_config.reduction,
        loss_collection=None)
  else:
    distances = distance_fn(
        labels=sources,
        predictions=targets,
        weights=weights,
        reduction=distance_config.reduction,
        loss_collection=None)
    if sum_over_axis is not None and _is_reduced_by_average(
        distance_config.reduction):
      # The distance is divided by the size of targets tensor, so we need to
      # rescale the distance by multiplying the size of axis. Note, the distance
      # function with `axis` as a required argument (e.g., consine distance)
      # does not need to be rescaled.
      weights = tf.convert_to_tensor(value=weights)
      weights_shape = weights.get_shape().as_list()
      if weights_shape and weights_shape[sum_over_axis] != 1:
        raise ValueError('Shape of weights along the axis %d must be 1.' %
                         sum_over_axis)
      distances *= sources.shape.dims[sum_over_axis].value
  return distances
```

参数表：

- `sources` : Tensor ( float32 或 float64 )
- `targets` : Tensor (与 `sources` 类型保持一致)
- `weights` : (可选) 维度为 0 或 `targets` 与维度一致，必须保证可以广播到 `targets`
- `distance_config` : 计算距离的配置（或超参数） `nsl.configs.DistanceConfig` 实例包含以下配置：
  - `distance_type` :要应用的距离函数的类型。
    - `DistanceType.L1`
    - `DistanceType.L2`
    - `DistanceType.COSINE`
    - `DistanceType.JENSEN_SHANNON_DIVERGENCE`
    - `DistanceType.KL_DIVERGENCE`
  - `reduction` :距离 reduce 的方式。
    - `Reduction.AUTO`
    - `Reduction.NONE`
    - `Reduction.SUM`
    - `Reduction.SUM_OVER_BATCH_SIZE`
  - `sum_over_axis` :（可选）距离是沿指定轴的差值之和。注：该参数如果不是 `None` 并且 `weights` 的维度不为零，则 `weights` 沿着 `sum_over_axis` 的维度必须为 1。
  - `transform_fn` :（可选）如果设置，则 `sources` 和 `targets` 都将在计算距离之前进行转换。如果设置为`SOFTMAX`，它将在 `sum_over_axis` 指定的轴上执行，如果未指定轴，则为 -1。如果是 `None` ，将使用默认距离配置。

整体逻辑为：

- 加载距离配置 `distance_config`，如果是 `None`，则使用默认配置。
- 加载距离配置中参数 `distance_config.transform_fn`，如果不是 `None`，则分别对 `sources` 和 `targets` 应用 `_apply_transform`。
- 加载距离配置中参数 `distance_config.distance_type`，如果不是 `None`，则分别对 `sources` 和 `targets` 应用 `tf.nn.l2_normalize`。
- 判断求解的距离类型，对应使用 `distance_fn` 函数（这里不展开说明）。

# 四、对比分析

## 共同点

- 都能实现计算两组张量之间的距离 API 的基本功能，都可以计算 1 范数、2 范数；
- 对于输入维度不同的两组张量，都可以进行广播机制（broadcastable）；
- API 都可以对指定维度（轴）进行计算。

## 不同点

- PyTorch 可以实现任意阶的 p 范数$p \in (-inf, inf)$，TensorFlow 只可以实现 l1 、l2 范数，但是还包含了其他的距离求解方式，例如`COSINE`（余弦距离）、`JENSEN_SHANNON_DIVERGENCE` （JS 散度）、`KL_DIVERGENCE` （KL 散度）；
- PyTorch 的 API 有参数 `eps` :避免出现除零错误， TensorFlow 则是隐式的避免了错误
- PyTorch 的 API 的参数 `keepdim`：是否保持张量维度，对应 TensorFlow 中的 reduce 操作。keepdim 可由 reduction 参数实现。
- TensorFlow 的 API 有参数 `weights` （权重）， PyTorch 并没有相关参数。

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.nn.functional.pairwise_distance(
  x: Tensor,
  y: Tensor,
  p: float=2.0,
  epsilon: float=1e-6,
  keepdim: bool=False,
  name: str=None)
```

注：其中参数名使用 `epsilon` 为了与 Paddle 其他 API 参数名保持一致。

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `Paddle/python/paddle/nn/functional/distance.py`（目前尚无该文件，故需要新建）。
Paddle 中已有 `nn.PairwiseDistance` class API 的具体实现逻辑，位于其 `forward` 函数中，因此只需把该 API 的计算逻辑提取出来，用于实现 `paddle.nn.functional.pairwise_distance` API。
class API 中的具体实现：

- 通过 `in_dygraph_mode` 判断是否为新动态图，如果是，则先通过调用 `_C_ops.elementwise_sub` 逐元素相减，然后调用 `_C_ops.final_state_p_norm` 函数计算两组张量间的距离范数；
- 通过 `_in_legacy_dygraph` 判断是否为旧动态图，如果是，则先通过调用 `_C_ops.elementwise_sub` 逐元素相减，然后则调用 `_C_ops.p_norm` 计算两组张量间的距离范数。
- 如果是静态图的话，首先实例化 `LayerHelper("PairwiseDistance", name=self.name)` ，然后调用 `paddle.subtract` 逐元素相减，调用`helper.append_op`加载参数， 调用`helper.create_variable_for_type_inference` 计算两组张量间的距离范数。

经测试，输入 x，y 的 shape 皆为(D, )，调用 `nn.PairwiseDistance` class API 进行计算时会报维度错误，因为调用 `_C_ops.p_norm` 或 `_C_ops.final_state_p_norm` 时，维度参数为 1。

# 六、测试和验收的考量

测试考虑的 case 如下：

- 参数 `p` 各个取值的正确性：
  - 对于 0、 1、2 等任意正实数 p 范数能够计算对应的范数及正确的结果；
  - `-inf` 和 `inf` 能够计算对应的范数及正确的结果；
- 参数 `epsilon` 的正确性，能够保证程序不出现除零错误；
- 参数 `keepdim` 为 `True` 或者 `False` 的输出 shape 正确性；
- x 和 y 的形状为 (N, D) 或者 (D, ), 当 x 和 y 分别取以下四种情况： (N, D) 和 (D, )、(N, D) 和 (N, D)、(D, ) 和 (N, D) 以及 (D, ) 和 (D, )，该 API 是否可以计算得到正确的值和 shape；
- 在动态图、静态图下的都能得到正确的结果。

# 七、可行性分析及规划排期

`paddle.nn.functional.pairwise_distance` 从 `paddle.nn.PairwiseDistance` 提取并修改得到。

具体规划为

- 阶段一：提取 `nn.PairwiseDistance` 主要逻辑到 `nn.functional.pairwise_distance`，在 `nn.PairwiseDistance` 中调用它，保证其逻辑不变。
- 阶段二：完成 `nn.functional.pairwise_distance` 单元测试
- 阶段三：该 API 书写中英文档

# 八、影响面

增加了一个 `nn.functional.pairwise_distance` API，并对原有的 `nn.PairwiseDistance` class API 进行修改，使其直接调用 `nn.functional.pairwise_distance` API，与 Layer 文件夹下的其他 class API 书写方式一致。

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.nn.functional.pairwise_distance](https://pytorch.org/docs/stable/generated/torch.nn.functional.pairwise_distance.html?highlight=pairwise#torch.nn.functional.pairwise_distance)

[torch.nn.PairwiseDistance](https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html#torch.nn.PairwiseDistance)

## TensorFlow

[nsl.lib.pairwise_distance_wrapper](https://www.tensorflow.org/neural_structured_learning/api_docs/python/nsl/lib/pairwise_distance_wrapper)

## Paddle

[paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/norm_cn.html#norm)

[paddle.dist](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist)

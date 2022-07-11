# paddle.bucketize 设计文档

| API 名称     |                paddle.bucketize           |
| ------------ | ---------------------------------------- |
| 提交作者     | PommesPeter                               |
| 提交时间     | 2022-07-09                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本  | develop                                   |
| 文件名       | 20220709_api_design_for_bucketize.md      |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算相关 API，Paddle 需要扩充 API `paddle.bucketize`。

## 2、功能目标

增加 API `paddle.bucketize`，用于根据 `sorted_sequence` 序列计算出 `x` 中每个元素的区间索引。

## 3、意义

为 Paddle 增加神经网络相关的距离计算函数，丰富 `paddle` 中科学计算相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `bucketize` API，但是存在 `searchsorted` API，参考其他框架可以发现，没有专门针对一维 `sorted_sequence` 进行计算的 api，直接使用 `searchsorted` API 导致花费时间在判断维度上。
- 该 API 的实现及测试主要参考目前 Paddle 中含有的 `paddle.searchsorted`。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `torch.bucketize` 的API，详细参数为 `torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None) → Tensor`。

在 PyTorch 中的介绍为：

> Returns the indices of the buckets to which each value in the `input` belongs, where the boundaries of the buckets are set by `boundaries`. Return a new tensor with the same size as `input`. If `right` is False (default), then the left boundary is closed. More formally, the returned index satisfies the following rules:
>
> | `right` | *returned index satisfies*                                |
> | ------- | --------------------------------------------------------- |
> | False   | `boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]` |
> | True    | `boundaries[i-1] <= input[m][n]...[l][x] < boundaries[i]` |

在实现方法上，PyTorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Bucketization.cpp)

实现代码：
```cpp
namespace at {
namespace native {

namespace {

// ...

}

// ...

Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt,
    Tensor& result) {

  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  bool is_right = side_opt ? *side_opt == "right" : right;

  if (self.numel() == 0) {
    return result;
  }

  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
    dispatch(out, self, sorted_sequence, out_int32, is_right, sorter);
  }
  else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    dispatch(out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor& bucketize_out_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_cpu(boundaries, self, out_int32, right, nullopt, nullopt, result);
  return result;
}

Tensor bucketize_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_cpu(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_cpu(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_cpu(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

}} // namespace at::native
```

参数表：

- input（Tensor or Scalar）：N-D Tensor，

- boundaries（Tensor）：，1-D Tensor，必须包含一个单调递增的序列。

- out_int32（bool，optional）：指明输出数据类型。如果是True，则输出torch.int32；如果是False，则输出torch.int64。默认是False。

- right（bool，optional）：如果为 False，返回找到的第一个合适的位置； 如果为 True，返回最后一个这样的索引； 如果没有找到合适的索引，则返回0作为非数值值(例如，Nan，Inf)或边界的大小（通过最后一个索引）。

  换句话说，如果为 False，则从边界获取输入中每个值的下界索引； 如果为 True，则获取上界索引。 默认值为 False。

- out（Tensor，optional）：输出的Tensor必须和输出的Tensor大小相同。

## Tensorflow

Tensorflow 中有 `tf.transform.bucketize` API，具体参数为 `tft.bucketize( x: common_types.ConsistentTensorType, num_buckets: int, epsilon: Optional[float] = None, weights: Optional[tf.Tensor] = None, elementwise: bool = False, name: Optional[str] = None) -> common_types.ConsistentTensorType`

在实现方法上，Tensorflow 是通过 Python API 的方式组合实现的，[代码位置](https://github.com/tensorflow/transform/blob/d0c3349403120a2cf1177c111b674c07e9b38398/tensorflow_transform/mappers.py#L1690-L1770)

代码实现：
```python
@common.log_api_use(common.MAPPER_COLLECTION)
def bucketize(x: common_types.ConsistentTensorType,
              num_buckets: int,
              epsilon: Optional[float] = None,
              weights: Optional[tf.Tensor] = None,
              elementwise: bool = False,
              name: Optional[str] = None) -> common_types.ConsistentTensorType:
  with tf.compat.v1.name_scope(name, 'bucketize'):
    if not isinstance(num_buckets, int):
      raise TypeError('num_buckets must be an int, got %s' % type(num_buckets))

    if num_buckets < 1:
      raise ValueError('Invalid num_buckets %d' % num_buckets)

    if isinstance(x, (tf.SparseTensor, tf.RaggedTensor)) and elementwise:
      raise ValueError(
          'bucketize requires `x` to be dense if `elementwise=True`')

    if epsilon is None:
      # See explanation in args documentation for epsilon.
      epsilon = min(1.0 / num_buckets, 0.01)

    x_values = tf_utils.get_values(x)
    bucket_boundaries = analyzers.quantiles(
        x_values,
        num_buckets,
        epsilon,
        weights,
        reduce_instance_dims=not elementwise)

    if not elementwise:
      return apply_buckets(x, bucket_boundaries)

    num_features = tf.math.reduce_prod(x.get_shape()[1:])
    bucket_boundaries = tf.reshape(bucket_boundaries, [num_features, -1])
    x_reshaped = tf.reshape(x, [-1, num_features])
    bucketized = []
    for idx, boundaries in enumerate(tf.unstack(bucket_boundaries, axis=0)):
      bucketized.append(apply_buckets(x_reshaped[:, idx],
                                      tf.expand_dims(boundaries, axis=0)))
    return tf.reshape(tf.stack(bucketized, axis=1),
                      [-1] + x.get_shape().as_list()[1:])
```

参数表：

| Args          |                                                              |
| :------------ | ------------------------------------------------------------ |
| `x`           | 一个数字输入的 `Tensor`或`CompositeTensor`，其值应被映射到桶中。对于一个`CompositeTensor`，只有非缺失的值才会被包括在定量计算中，`bucketize`的结果将是一个`CompositeTensor`，其非缺失的值被映射到桶中。如果 elementwise=True，那么`x`必须是密集的。 |
| `num_buckets` | 输入的`x`中的值被分成大小大致相等的桶，桶的数量是`num_buckets`。 |
| `epsilon`     | （可选）误差容限，通常是一个接近于零的小部分。如果调用者没有指定一个值，将根据实验结果计算出一个合适的值。对于小于 100 的`num_buckets`，选择 0.01 的值来处理高达约 1 万亿的输入数据值的数据集。如果`num_buckets`更大，那么 epsilon 被设置为 (1 / `num_buckets`) 以执行更严格的误差容忍度，因为更多的桶将导致每个桶的范围更小，所以我们希望边界不那么模糊。详情见analyzers.quantiles()。 |
| `weights`     | （可选）用于定量的权重张量。张量必须与 x 具有相同的形状。    |
| `elementwise` | （可选）如果为真，对 tensor 的每个元素进行独立的桶化。       |
| `name`        | (可选) 该操作的名称。                                        |

# 四、对比分析

## 共同点

- 都能实现根据 `sorted_sequence` 计算出输入 `x` 中每个元素所对应的区间索引

## 不同点

- PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。
- PyTorch 输入参数比较简单，可选的操作比较少。
- Tensorflow 则是通过 Python API 直接实现其对应的功能。
- Tensorflow 有 `num_buckets`、`epsilon`、`weights` 等参数的设置，可调整的程度更高。


# 五、设计思路与实现方案

## 命名与参数设计

添加 API

```python
paddle.bucketize(
    x: Tensor,
    sorted_sequence: Tensor,
    out_int32: bool=False,
    right: bool=False,
    name: str=None
)
```

## 底层 OP 设计

使用已有的 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 实现于 `python/paddle/tensor/search.py`

首先，`bucketize` 主要针对一维情况下的 `sorted_sequence`，所以需要对输入的维度大小进行判断，通过断言进行判断，当输入维度不为 1 时触发 `AssertError`。

随后，Paddle 中已有 `searchsorted` API 的具体实现逻辑，位于 `python/paddle/tensor/search.py` 下的 `searchsorted` 函数中，因此只需要调用其函数即可。

# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性，使用 numpy 作为参考标准
- 参数 `right` 为 True 和 False 时输出的正确性
- 参数 `out_int32` 为 True 和 False 时 dtype 输出的正确性
- 参数 `x` 类型的正确性，若类型不为 Tensor 则抛出异常
- 参数 `sorted_sequence` 的维度正确性，该 API 只针对 `sorted_sequence` 是一维的情况，所以对于输入需要约束
- 未输入 `right` 时的输出正确性；
- 未输入 `out_int32` 时的输出正确性；

# 七、可行性分析和排期规划

方案主要依赖现有 Paddle API 组合而成，且依赖的 `paddle.searchsorted` 已经在 Paddle repo 的 [python/paddle/tensor/search.py](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/python/paddle/tensor/search.py#L910)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块是否有影响

# 名词解释

无

# 附件及参考资料

## PyTorch

[torch.bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html)

[torch.searchsorted](https://pytorch.org/docs/stable/generated/torch.searchsorted.html?highlight=searchsorted#torch.searchsorted)

## tensorflow

[tf.transform.bucketize](https://www.tensorflow.org/tfx/transform/api_docs/python/tft/bucketize)

[tf.searchsorted](https://www.tensorflow.org/api_docs/python/tf/searchsorted)

## Paddle

[paddle.searchsorted](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/searchsorted_cn.html)
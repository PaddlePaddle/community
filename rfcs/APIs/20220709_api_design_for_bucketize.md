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
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

/* Implement a numpy like searchsorted and a TF like bucketize function running on cpu
 *
 * - torch.searchsorted(sorted_sequence, values, right=False, side='left', out_int32=False, sorter=None)
 *   sorted_sequence - N*D or 1D (apply to all values) tensor containing sorted sequences in last dimension
 *   values          - N*D tensor or a Scalar (when sorted_sequence is 1D) containing the search values
 *   right           - corresponding to lower bound if False and upper bound if True
 *   side            - (preferred to right) corresponding to lower bound if 'left' and upper bound if 'right'
 *   out_int32       - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *   sorter          - if provided, sorted_sequence may not be sorted and the sorted order is given by this tensor
 *
 * - torch.bucketize(values, boundaries, right=False, out_int32=False)
 *   values     - N*D tensor or a Scalar containing the search value
 *   boundaries - 1D tensor containing a sorted sequences
 *   right      - corresponding to lower bound if False and upper bound if True
 *   out_int32  - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *
 * - Restrictions are defined in searchsorted_pre_check()
 */

namespace at {
namespace native {

namespace {

// minimal size for searchsorted_cpu_contiguous to run parallel (multithread)
constexpr int64_t SEARCHSORTED_GRAIN_SIZE = 200;

// customized lower_bound func to ensure the low bound of 'nan', 'inf' etc. be the end of boundary
// and we can properly handle a sorter argument
// std::lower_bound can not be used here since its customized comparator need strict weak ordering
// and the customized comparators require both arguments to have the same type, which wouldn't
// happen when comparing val of input_t to an indexer value from sorter of int64
template<typename input_t>
int64_t cus_lower_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

// customized upper_bound func to ensure we can properly handle a sorter argument
// std::upper_bound can not be used here since its customized comparator requires both arguments to have the
// same type, which wouldn't happen when comparing val of input_t to an indexer value from sorter of int64
template<typename input_t>
int64_t cus_upper_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right, const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();
  const int64_t *data_st = sorter.defined() ? sorter.data_ptr<int64_t>() : nullptr;
  output_t *data_out = result.data_ptr<output_t>();

  bool is_1d_boundaries = boundaries.dim() == 1;
  at::parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      // If boundaries tensor is 1d, we always search the entire boundary tensor
      int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
      int64_t end_bd = start_bd + idim_bd;

      int64_t pos = !right ?
        cus_lower_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd :
        cus_upper_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd;

      // type conversion might happen here
      data_out[i] = pos;
    }
  });
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries, bool out_int32, bool right, const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out_cpu",
        [&] {
          searchsorted_cpu_contiguous<scalar_t, int64_t>(
              result, input, boundaries, right, sorter);
        });
  }
  else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out_cpu",
        [&] {
          searchsorted_cpu_contiguous<scalar_t, int>(
              result, input, boundaries, right, sorter);
        });
  }
}

}

Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to opposites
  bool is_right = side_opt ? *side_opt == "right" : right;

  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy so we can later copy back, maintaing the original result tensor
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

  // if result is non-contiguous, we wrote the answer to a copied version, so we copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor searchsorted_cpu(
      const Tensor& sorted_sequence,
      const Tensor& self,
      bool out_int32,
      bool right,
      const c10::optional<c10::string_view> side_opt,
      const c10::optional<Tensor>& sorter_opt) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_cpu(sorted_sequence, self, out_int32, right, side_opt, sorter_opt, result);
  return result;
}

Tensor searchsorted_cpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_cpu(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt);
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
  """Returns a bucketized column, with a bucket index assigned to each input.
  Args:
    x: A numeric input `Tensor` or `CompositeTensor` whose values should be
      mapped to buckets.  For a `CompositeTensor` only non-missing values will
      be included in the quantiles computation, and the result of `bucketize`
      will be a `CompositeTensor` with non-missing values mapped to buckets. If
      elementwise=True then `x` must be dense.
    num_buckets: Values in the input `x` are divided into approximately
      equal-sized buckets, where the number of buckets is `num_buckets`.
    epsilon: (Optional) Error tolerance, typically a small fraction close to
      zero. If a value is not specified by the caller, a suitable value is
      computed based on experimental results.  For `num_buckets` less than 100,
      the value of 0.01 is chosen to handle a dataset of up to ~1 trillion input
      data values.  If `num_buckets` is larger, then epsilon is set to
      (1/`num_buckets`) to enforce a stricter error tolerance, because more
      buckets will result in smaller range for each bucket, and so we want the
      boundaries to be less fuzzy. See analyzers.quantiles() for details.
    weights: (Optional) Weights tensor for the quantiles. Tensor must have the
      same shape as x.
    elementwise: (Optional) If true, bucketize each element of the tensor
      independently.
    name: (Optional) A name for this operation.
  Returns:
    A `Tensor` of the same shape as `x`, with each element in the
    returned tensor representing the bucketized value. Bucketized value is
    in the range [0, actual_num_buckets). Sometimes the actual number of buckets
    can be different than num_buckets hint, for example in case the number of
    distinct values is smaller than num_buckets, or in cases where the
    input values are not uniformly distributed.
    NaN values are mapped to the last bucket. Values with NaN weights are
    ignored in bucket boundaries calculation.
  Raises:
    TypeError: If num_buckets is not an int.
    ValueError: If value of num_buckets is not > 1.
    ValueError: If elementwise=True and x is a `CompositeTensor`.
  """
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

- 数值结果的一致性，使用 numpy 作为参考标准
- 参数 `right` 为 True 和 False 时输出的正确性
- 参数 `out_int32` 为 True 和 False 时 dtype 输出的正确性；
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
# paddle.[Tensor.]cummax 设计文档

| API名称                                                      | paddle.[Tensor.]cummax        |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | jinyouzhi
                     |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-16                      |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0                              |
| 文件名                                                       | 20220316_api_design_for_cummax.md<br> |

# 一、概述

## 1、相关背景

cummax 是指求累积最大值（cumulative max）的功能。即求
$$
    y_i = \max\(x_1, x_2, x_3, \cdots , x_i\)
$$

PyTorch 和 Pandas 提供了该算子。

## 2、功能目标

cummax API 是一个按轴寻找累计最大值和最大值所在位置的 API。此任务的目标是在 Paddle 框架中，新增 cummax API，调用路径为：`paddle.cummax`和 `paddle.Tensor.cummax`。
## 3、意义

完善矩阵运算的基本功能，增强统计运算完善度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有 API（https://pytorch.org/docs/stable/generated/torch.cummax.html）

在 PyTorch 文档中，介绍为：

```
Returns a namedtuple (values, indices) where values is the cumulative maximum of elements of input in the dimension dim. And indices is the index location of each maximum value found in the dimension dim.

Parameters
 - input (Tensor) – the input tensor.
 - dim (int) – the dimension to do the operation over

Keyword Arguments
 - out (tuple, optional) – the result tuple of two output tensors (values, indices)
```
即输入参数为 Tensor 和指定的维，两个值和索引的切片。

### 实现方法

在实现方法上, PyTorch 通用实现采用的遍历，[CPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp#L638)，CUDA 采用的[GPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ScanKernels.cpp#L17)。
核心代码为：
CPU:
```cpp
template<typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1* self_data, T1* values_data, T2* indices_data,
          int self_dim_size, int self_stride, int values_stride, int indices_stride) {
      Operation op;
      T1 out = self_data[0];
      int idx = 0;
      for (const auto i : c10::irange(self_dim_size)) {
        T1 curr_elem = self_data[i*self_stride];
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            out = self_data[i*self_stride];
            idx = i;
        }
        values_data[i*values_stride] = out;
        indices_data[i*indices_stride] = idx;
      }
}

void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16,
    self.scalar_type(), "cummax_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummax_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    at::native::resize_output(values, self.sizes());
    at::native::resize_output(indices, self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummax_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummax(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummax_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}
```
GPU:
```cpp
void cummax_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummax_cuda_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}
```
```cpp
void launch_cummax_cuda_kernel(const TensorBase& self, const TensorBase& values, const TensorBase& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16,
    self.scalar_type(), "cummax_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? (-1*std::numeric_limits<scalar_t>::infinity()) : std::numeric_limits<scalar_t>::lowest();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::greater_equal<scalar_t>());
  });
}
```


## Pandas

Pandas 也提供了该 API [pandas.DataFrame.cummax¶
](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cummax.html#pandas-dataframe-cummax)。

介绍为：

```
Return cumulative maximum over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative maximum.

Parameters
axis{0 or ‘index’, 1 or ‘columns’}, default 0
The index or the name of the axis. 0 is equivalent to None or ‘index’.

skipnabool, default True
Exclude NA/null values. If an entire row/column is NA, the result will be NA.

*args, **kwargs
Additional keywords have no effect but might be accepted for compatibility with NumPy.

Returns
Series or DataFrame
Return cumulative maximum of Series or DataFrame.
```
值得注意的是，Pandas 提供了更多的附加选项，即对 NaN 值的处理方式。
并且 Pandas 支持 DataFrame 的操作。


### 实现方法

基于 Python 开发朴素的循环，[源码](https://github.com/pandas-dev/pandas/blob/48d515958d5805f0e62e34b7424097e5575089a8/pandas/_libs/groupby.pyx#L1514)
其中 Pandas 的矩阵操作是基于 Numpy 开发的，直接以索引方式操作。

```python
    N, K = (<object>values).shape
    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue
            for j in range(K):

                if not skipna and na_possible and seen_na[lab, j]:
                    if uses_mask:
                        result_mask[i, j] = 1
                        # Set to 0 ensures that we are deterministic and can
                        #  downcast if appropriate
                        out[i, j] = 0

                    else:
                        out[i, j] = na_val
                else:
                    val = values[i, j]

                    if uses_mask:
                        isna_entry = mask[i, j]
                    else:
                        isna_entry = _treat_as_na(val, is_datetimelike)

                    if not isna_entry:
                        mval = accum[lab, j]
                        if compute_max:
                            if val > mval:
                                accum[lab, j] = mval = val
                        else:
                            if val < mval:
                                accum[lab, j] = mval = val
                        out[i, j] = mval
                    else:
                        seen_na[lab, j] = 1
                        out[i, j] = val

```

# 四、对比分析

比较分析：在基于 CPU 的方案上思路较为一致，其中 PyTorch 对于矩阵的操作是基于 stride 和指针完成的，Pandas 基于 Numpy 提供的矩阵操作能力，所以以索引方式操作。
PyTorch 还提供了基于 CUDA 的算子实现。
评价：Pandas 的矩阵操作实际由 Numpy 支撑，在该运算实现效率上应不如 PyTorch 实现的；在功能上，Pandas 支持了可选的 NaN 值处理选项，有一定灵活性。

# 五、方案设计

## 命名与参数设计

API设计为`paddle.cummax(x, axis , dtype, name)`以及`paddle.Tensor.cumax(axis, dtype, name)`。参数设计参考`paddle.cumsum`。
- x (Tensor) - 需要进行累积最大值统计的 Tensor。
- axis (int, 可选) - 指明需要统计的维度。-1代表最后一维。默认：None，将输入展开为一维变量再进行累加计算。
- dtype (str，可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64. 如果指定了，那么在执行操作之前，输入张量将被转换为dtype. 这对于防止数据类型溢出非常有用。默认为：None。
- name  (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 Name 。

## 底层OP设计

参考 PyTorch 实现。

## API实现方案

主要参考 PyTorch 的计算逻辑开发，算子的接口和调用逻辑参考`paddle.cumsum`设计。
Python 接口实现位置为`paddle/tesnor/math.py`。

# 六、测试和验收的考量

测试考虑的case如下：

- 动态图，静态图，与 PyTorch 的结果保持一致；
- NaN 处理：关注异常值，边界情况的处理；
- 错误检查：输入输出矩阵形状的检查。

# 七、可行性分析及规划排期

3/20 完成技术方案；
3/25 开发 Python 实现代码 & 英文 API文档；
3/30 开发 C++ 实现代码 & CUDA 实现代码。

# 八、影响面

为独立新增API，对其他模块没有影响

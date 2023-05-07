# paddle.[Tensor.]cummin 设计文档

| API名称                                                      | paddle.[Tensor.]cummin          |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者   | NetPunk                       |
| 提交时间| 2023-03-02                           |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名                                                       | 20230302_api_design_for_cummin.md |

# 一、概述

## 1、相关背景

cummin函数的功能为求累积最小值（cumulative min）。对于输入向量/矩阵，第i个位置的计算方式为：
$$
y_i = \min(x_1, x_2, x_3, \cdots , x_i)
$$

PyTorch、NumPy 和 Pandas 提供了相似算子。

## 2、功能目标

cummin API 是一个按轴寻找累计最小值和最小值所在位置的 API。此任务的目标是在 Paddle 框架中，新增 cummin API，调用路径为：`paddle.cummin`和 `paddle.Tensor.cummin`。
## 3、意义

完善矩阵运算的基本功能，增强统计运算完善度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有 [API](https://pytorch.org/docs/stable/generated/torch.cummin.html)

在 PyTorch 文档中，介绍为：

```
Returns a namedtuple (values, indices) where values is the cumulative minimum of elements of input in the dimension dim. And indices is the index location of each minimum value found in the dimension dim.

Parameters
 - input (Tensor) – the input tensor.
 - dim (int) – the dimension to do the operation over

Keyword Arguments
 - out (tuple, optional) – the result tuple of two output tensors (values, indices)
```
输入数据Tensor和cummin操作的维度dim，输出一个tuple包含计算结果values和索引indices

### 实现方法

在实现方法上, PyTorch采用的CPU实现为：循环遍历赋值，而CUDA实现则是调用pytorch自己实现的scan_with_indices函数。
核心代码为如下
[CPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp#L769):

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

void cummin_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16,
    self.scalar_type(), "cummin_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::less_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummin_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
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
      at::_cummin_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummin(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummin_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}
```
[GPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ScanKernels.cpp#L45):

```cpp
void cummin_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummin_cuda_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}
```
```cpp
void launch_cummin_cuda_kernel(const TensorBase& self, const TensorBase& values, const TensorBase& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16,
    self.scalar_type(), "cummin_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? std::numeric_limits<scalar_t>::infinity() : std::numeric_limits<scalar_t>::max();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::less_equal<scalar_t>());
  });
}
```

~~~cpp
template<typename scalar_t, typename BinaryFunction>
void scan_dim_with_indices(const TensorBase& self, const TensorBase& values, const TensorBase& indices, //int64_t dim) {
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(values.is_contiguous() && indices.is_contiguous());
  if (dim == ndim - 1) {
    scan_innermost_dim_with_indices<scalar_t>(*self_, values, indices, init, binary_op);
  } else {
    scan_outer_dim_with_indices<scalar_t>(*self_, values, indices, dim, init, binary_op);
  }
}
~~~

其中函数`scan_innermost_dim_with_indices`和`scan_outer_dim_with_indices`的相关代码较长，它们的功能是在不同维度上对输入进行并行的累积操作，其中关于并行扫描部分实现的代码值得参考。

CPU/GPU反向计算

~~~cpp
Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input, const Tensor& indices, int64_t dim) {
  if (input.numel() == 0) {
    return input;
  }
  auto result = at::zeros(input.sizes(), input.options());

  // for composite compliance, use out-of-place variant of
  // `scatter_add` if `indices` or `grad` is a Tensor Subclass.
  if (areAnyTensorSubclassLike({indices, grad})) {
    return result.scatter_add(dim, indices, grad);
  }
  return result.scatter_add_(dim, indices, grad);
}
~~~

## NumPy

NumPy 具有相似功能的 API 是 `numpy.minimum.accumulate()`，文档参见 [numpy.ufunc.accumulate — NumPy v1.22 Manual](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.accumulate.html)。

NumPy 的策略是提供一种更具兼容性的实现方式，组合实现该功能（尽管其单独提供了 `cumsum` 和 `cumprod`，但没有单独提供 `cummin`），有别于 PyTorch 分别 native 实现 `cumprod`、`cumsum` 以及 `cummin`。

## Pandas

Pandas 也提供了该 API [pandas.DataFrame.cummin](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cummin.html)。

介绍为：

```
Return cumulative minimum over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative minimum.

Parameters
axis{0 or ‘index’, 1 or ‘columns’}, default 0
The index or the name of the axis. 0 is equivalent to None or ‘index’.

skipnabool, default True
Exclude NA/null values. If an entire row/column is NA, the result will be NA.

*args, **kwargs
Additional keywords have no effect but might be accepted for compatibility with NumPy.

Returns
Series or DataFrame
Return cumulative minimum of Series or DataFrame.
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

比较分析：不同框架在基于 CPU 的方案上思路较为一致。其中 PyTorch 对于矩阵的操作是基于 stride 和指针完成的；Pandas 基于 Numpy 提供的矩阵操作能力，所以以索引方式操作；NumPy 没有原生实现该功能。
PyTorch 还提供了基于 CUDA 的算子实现。
评价：Pandas 的矩阵操作实际由 Numpy 支撑，在该运算实现效率上应不如 PyTorch 实现的；在功能上，Pandas 支持了可选的 NaN 值处理选项，有一定灵活性；NumPy 没有提供原生的 `cummin` 实现，而是基于组合的方式。
就基于已有的方法组合实现这一途径，经过调研，PaddlePaddle 和 PyTorch 都已原生实现 `cumsum` 和 `cumprod`，为 `cummin` 提供原生实现，应能够提供更好的性能。

# 五、方案设计

## 命名与参数设计

API设计为`paddle.cummin(x, axis, dtype, name)`以及`paddle.Tensor.cummin(axis, dtype, name)`。

paddle.cummin
----------------------
参数
:::::::::
- x (Tensor) - 累积最小值的输入，需要进行累积最小值操作的 Tensor。
- axis (int, 可选) - 指明需要统计的维度。-1代表最后一维。默认：None，将输入展开为一维变量再进行累积最小值计算。
- dtype (str，可选) - 指定输出索引的数据类型，可以为int32和int64，默认：int64。
- name (str，可选) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
返回
:::::::::
- Out (tuple) - 返回累积最小值结果和对应的索引信息。累积最小值结果的数据类型和输入`x`一致。

paddle.Tensor.cummin指向paddle.cummin，两者是相同的API

## 底层OP设计

cpu：
前向计算，需要计算cummin结果Out和对应的Indices，没有在paddle内部找到可以直接计算Indices的API可供调用，因此需要实现一个能够同时计算cmmin和Indices的函数ScanWithIndicesKernel
后向计算，调用cpu_scatter_add函数在Indices指定位置分配grad值，具体可以查看上面的pytorch实现

gpu：
前向计算，大体过程与cumsum类似，但是在计算部分需要实现一个能够同时计算cummin和Indices的函数ScanWithIndicesKernel
后向计算，调用gpu_scatter_add函数在Indices指定位置分配grad值，具体可以查看上面的pytorch实现

前向函数定义

~~~cpp
template <typename T, typename Context>
void CumminKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  int dtype,
                  DenseTensor* out,
                  DenseTensor* indices);
~~~

后向函数定义

~~~cpp
template <typename T, typename Context>
void CumminGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      int axis,
                      int dtype,
                      DenseTensor* x_grad);
~~~

## API实现方案

主要参考 PyTorch 的计算逻辑开发，算子的接口和调用逻辑参考`paddle.cumsum`设计。
Python 接口实现位置为`paddle/tesnor/math.py`。

# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算和反向计算；
  - axis 维度：0，1，默认（None），-1等；
  - 计算dtype类型：验证 `float64`，`int32`等；
  - 索引dtype类型：验证指定索引数据类型是否正确，测试`int64`和`int32`

- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- API调用和OP计算分别测试
- 错误检查：输入参数类型、形状的有效性校验。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

1st week：中英文 API 文档编写 & 测试样例；

2nd week：CPU 后端 C++ 实现和前端 Python 代码；

3rd week：CUDA 后端 C++ 实现和前端 Python 代码；

4th week：测试和完善文档。

# 八、影响面

为独立新增API，对其他模块没有影响
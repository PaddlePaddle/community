# tensor_split / hsplit / dsplit API 设计文档

| API 名称 | tensor_split / hsplit / dsplit |
| - | - |
| 提交作者 | megemini(柳顺) |
| 提交时间 | 2023-10-03 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20231003_api_design_for_tensor_split.md |

# 一、概述

## 1、相关背景

在处理图像、文本等数据时，往往需要对输入的 Tensor 进行拆分，比如将一个 batch 的图像拆分为两个或者多个，这时候便可以使用 `tensor_split` 函数。

另外，还有 `hsplit` 函数，即在水平轴上进行拆分，`dsplit` 函数，即在深度轴上进行拆分。

目前 `Paddle` 框架中没有 `tensor_split`、`hsplit`、`dsplit` 函数，特在此任务中实现，涉及接口为：

- `paddle.tensor_split` 作为独立的函数调用
- `Tensor.tensor_split` 作为 Tensor 的方法使用
- `paddle.hsplit` 作为独立的函数调用
- `Tensor.hsplit` 作为 Tensor 的方法使用
- `paddle.dsplit` 作为独立的函数调用
- `Tensor.dsplit` 作为 Tensor 的方法使用

以提升飞桨 API 的丰富程度。

## 2、功能目标

将一个 Tensor 按给定的轴和信息切分成多个子 Tensor，这些子 Tensor 是原 Tensor 的 view，该操作称为 `tensor_split`。此外，其在特定轴上时的行为，称作 `hsplit`、 `dsplit`。调用路径为：

- `paddle.tensor_split` 作为独立的函数调用
- `Tensor.tensor_split` 作为 Tensor 的方法使用
- `paddle.hsplit` 作为独立的函数调用
- `Tensor.hsplit` 作为 Tensor 的方法使用
- `paddle.dsplit` 作为独立的函数调用
- `Tensor.dsplit` 作为 Tensor 的方法使用

## 3、意义

为 `Paddle` 增加 `tensor_split`、`hsplit`、`dsplit` 操作，丰富 `Paddle` 中张量视图的 API。

# 二、飞桨现状

目前 `Paddle` 在 python 端缺少相关接口的实现，而在底层也没有相关算子。

`python/paddle/tensor/manipulation.py` 文件中实现了若干对于 `Tensor` 操作的接口，如：

- `split` 拆分 Tensor
- `vsplit` 垂直轴拆分 Tensor
- `unbind` 移除轴并拆分 Tensor
- `chunk` 拆分 Tensor
- `slice` 沿多个轴生成 Tensor 切片

这些接口都可以实现拆分 Tensor，不过每个接口的输入参数不尽相同。

另外，对于 `split` 函数与 `tensor_split` 的区别，这里引用 [Pytorch文档学习 TORCH.TENSOR_SPLIT](https://blog.csdn.net/Jamesgender/article/details/130559738) ：

> 这个方法和 split 方法长得很像。他们的作用都是根据 indices_or_sections，把输入拆分成几个视图。区别在于：
> - split 的 indices_or_sections 的类型为 int 或者 list ^*^，而 tensor_split 的类型为 tensor，int，list，tuple of list。
> - 当 indices_or_sections 类型为 int 时，对于 split，其按照 int 的值 n 不断划分，直到最后一个取不完分为最后一组。对于 tensor_split，会优先计算第一个分组，大小为 int(size/n)+1，然后剩下的分组大小尽可能一样。
> - 当 indices_or_sections 类型为 list 时，对于 split，划分值之和不得超过元素个数，比如不能写成【3,4,15】，否则会报错。对于 tensor_split，其划分的方法是对 list 或者 tuple 中的元素，根据元素组成的区间进行划分。如对【3,4,15】，下标小于3的，下标小于4大于3的，下标小于6大于4的，下标小于15大于6的，该方法会划分出以上4组。下标超出实际大小的时候会返回一个空的 tensor。
> ``` python
> x = torch.arange(7)
> print(torch.tensor_split(x, 3))
> print(torch.split(x, 3))
>
> (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
> (tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6]))
> 
> x = torch.arange(7)
> print(torch.tensor_split(x, (3,4,15)))
> print(torch.split(x, [3,4]))
> 
> (tensor([0, 1, 2]), tensor([3]), tensor([4, 5, 6]), tensor([], dtype=torch.int64))
> (tensor([0, 1, 2]), tensor([3, 4, 5, 6]))
> ```
> 

^*^ 注 : `Paddle` 的 `split` 函数签名为 `split(x, num_or_sections, axis=0, name=None)`，与上文中介绍的不一样，但并不影响后续的分析。

可以利用已有的这些接口构造 `tensor_split` 等方法。


# 三、业内方案调研

## PyTorch

`PyTorch` 底层通过 c++ 实现 `tensor_split`、`hsplit`、`dsplit` 函数，并通过上层的 python 对外开放相应接口。

相应文档：

- [TORCH.TENSOR_SPLIT](https://pytorch.org/docs/stable/generated/torch.tensor_split.html?highlight=tensor_split)
- [TORCH.HSPLIT](https://pytorch.org/docs/stable/generated/torch.hsplit.html?highlight=hsplit)
- [TORCH.DSPLIT](https://pytorch.org/docs/stable/generated/torch.dsplit.html?highlight=dsplit)

c++ 接口文件在：

- [TensorShape.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp) : aten/src/ATen/native/TensorShape.cpp

相应接口为：

- `torch.tensor_split(input, indices_or_sections, dim=0) → List of Tensors`

    - 文档描述
    > Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections. This function is based on NumPy’s numpy.array_split().

    - 参数列表
    > input (Tensor) – the tensor to split
    > indices_or_sections (Tensor, int or list or tuple of ints) 
    > dim (int, optional) – dimension along which to split the tensor. Default: 0

    - 返回值
    > output (List of Tensors)

    - 源码
    ``` cpp
    std::vector<Tensor> tensor_split_sections_symint(const Tensor& self, c10::SymInt sym_sections, int64_t dim) {
      TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
      int64_t dim_ = maybe_wrap_dim(dim, self.dim());
      // NB: intentional, sections specifies number of output tensors, which
      // cannot be polymorphic
      int64_t sections = sym_sections.guard_int(__FILE__, __LINE__);
      TORCH_CHECK(sections > 0, "number of sections must be larger than 0, got ", sections);
      const auto dim_size = self.sym_size(dim_);
      std::vector<Tensor> splits(sections);
      auto min_split_size = dim_size / sections;
      auto num_splits_one_extra = dim_size % sections;
      c10::SymInt start_idx = 0;
      for (const auto split_idx : c10::irange(sections)) {
        auto split_size = (num_splits_one_extra > split_idx) ? (min_split_size + 1) : min_split_size;
        splits[split_idx] = at::slice_symint(self, dim_, start_idx, start_idx + split_size);
        start_idx += split_size;
      }
      return splits;
    }

    template <typename T>
    std::vector<Tensor> _tensor_split_indices(const Tensor& self, ArrayRef<T> indices, int64_t dim) {
      TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
      int64_t dim_ = maybe_wrap_dim(dim, self.dim());
      int64_t num_indices = indices.size();
      std::vector<Tensor> splits(num_indices + 1);
      T start_idx(0);
      for (const auto split_idx : c10::irange(num_indices)) {
        auto end_idx = indices[split_idx];
        splits[split_idx] = at::symint::slice<T>(self, dim_, start_idx, end_idx);
        start_idx = end_idx;
      }
      splits[num_indices] = at::symint::slice<T>(self, dim_, start_idx, at::symint::size<T>(self, dim_));
      return splits;
    }

    std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim) {
      return _tensor_split_indices(self, indices, dim);
    }

    std::vector<Tensor> tensor_split_indices_symint(const Tensor& self, SymIntArrayRef indices, int64_t dim) {
      return _tensor_split_indices(self, indices, dim);
    }

    std::vector<Tensor> tensor_split(const Tensor& self, const Tensor& tensor_indices_or_sections, int64_t dim) {
      TORCH_CHECK(self.dim() > 0, "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ", self.dim()," dims");
      auto split_device = tensor_indices_or_sections.device();
      TORCH_CHECK(split_device == kCPU,
        "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on ", split_device);
      auto split_dtype = tensor_indices_or_sections.scalar_type();
      TORCH_CHECK(split_dtype == at::kLong,
        "tensor_split expected tensor_indices_or_sections to have dtype of long, but got ", split_dtype);
      auto split_dim = tensor_indices_or_sections.dim();
      TORCH_CHECK(split_dim == 1 || split_dim == 0,
        "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or one-dimensional tensor, but got a tensor with ", split_dim, " dims");

      if (split_dim == 0) {
        int64_t sections = tensor_indices_or_sections.item<int64_t>();
        return self.tensor_split(sections, dim);
      } else {
        auto indices_data = tensor_indices_or_sections.data_ptr<int64_t>();
        auto stride = tensor_indices_or_sections.stride(0);
        auto numel = tensor_indices_or_sections.numel();
        std::vector<int64_t> indices(numel);
        for (const auto offset : c10::irange(numel)) {
          // indices tensor could be non-contiguous
          indices[offset] = *(indices_data + offset * stride);
        }
        return self.tensor_split(indices, dim);
      }
    }
    ```

- `torch.hsplit(input, indices_or_sections) → List of Tensors`

    - 文档描述
    > Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections. Each split is a view of input.

    - 参数列表
    > input (Tensor) – tensor to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of Tensors)

    - 源码
    ``` cpp
    std::vector<Tensor> hsplit(const Tensor& self, int64_t split_size) {
      TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
      int64_t dim = (self.dim() == 1) ? 0 : 1;
      TORCH_CHECK(split_size != 0 && self.sym_sizes()[dim] % split_size == 0,
        "torch.hsplit attempted to split along dimension ", dim,", but the size of the dimension ", self.sizes()[dim], " is not divisible by the split_size ", split_size, "!");
      return at::tensor_split(self, split_size, dim);
    }

    std::vector<Tensor> hsplit(const Tensor& self, IntArrayRef split_sizes) {
      TORCH_CHECK(self.dim() >= 1, "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ", self.dim(), " dimensions!")
      return at::tensor_split(self, split_sizes, (self.dim() == 1) ? 0 : 1);
    }
    ```

- `torch.dsplit(input, indices_or_sections) → List of Tensors`

    - 文档描述
    > Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections. Each split is a view of input.

    - 参数列表
    > input (Tensor) – tensor to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of Tensors)

    - 源码
    ``` cpp
    std::vector<Tensor> dsplit(const Tensor& self, int64_t split_size) {
      TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
      TORCH_CHECK(split_size != 0 && self.sym_sizes()[2] % split_size == 0,
        "torch.dsplit attempted to split along dimension ", 2,", but the size of the dimension ", self.sizes()[2], " is not divisible by the split_size ", split_size, "!");
      return at::tensor_split(self, split_size, 2);
    }

    std::vector<Tensor> dsplit(const Tensor& self, IntArrayRef split_sizes) {
      TORCH_CHECK(self.dim() >= 3, "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ", self.dim(), " dimensions!")
      return at::tensor_split(self, split_sizes, 2);
    }    
    ```

可以看到，`hsplit`、`dsplit` 依赖于 `tensor_split` 函数的实现。

## TensorFlow

`TensorFlow` 并没有 `tensor_split` 函数，只实现了 `hsplit`、`dsplit` 函数。

相应文档：

- [tf.experimental.numpy.hsplit](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/hsplit?hl=en)
- [tf.experimental.numpy.dsplit](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/dsplit?hl=en)

`TensorFlow` 的 `hsplit`、`dsplit` 函数是一种 `Numpy` 实现的变体。

- [np_array_ops.py](https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/numpy_ops/np_array_ops.py) : python/ops/numpy_ops/np_array_ops.py

相应接口为：

- `tf.experimental.numpy.hsplit`

    - 文档描述
    > TensorFlow variant of NumPy's hsplit.

    - 参数列表
    > ary (Tensor) – tensor to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of Tensors)

- `tf.experimental.numpy.dsplit`

    - 文档描述
    > TensorFlow variant of NumPy's dsplit.

    - 参数列表
    > ary (Tensor) – tensor to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of Tensors)

上述两个函数的源码为：

``` python
def _split_on_axis(np_fun_name, axis):  # pylint: disable=missing-function-docstring
  @np_utils.np_doc(np_fun_name)
  def f(ary, indices_or_sections):
    # for 1-D array, hsplit becomes vsplit
    new_axis = np_utils.cond(
        math_ops.equal(axis, 1),
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(ary), 1), lambda: 0, lambda: axis
        ),
        lambda: axis,
    )
    if isinstance(indices_or_sections, int):
      ary_shape = ary.shape[new_axis]
      if ary_shape is not None and ary_shape % indices_or_sections:
        raise ValueError('array split does not result in an equal division')
    return split(ary, indices_or_sections, axis=new_axis)

  return f


vsplit = tf_export.tf_export('experimental.numpy.vsplit', v1=[])(
    _split_on_axis('vsplit', axis=0)
)
hsplit = tf_export.tf_export('experimental.numpy.hsplit', v1=[])(
    _split_on_axis('hsplit', axis=1)
)
dsplit = tf_export.tf_export('experimental.numpy.dsplit', v1=[])(
    _split_on_axis('dsplit', axis=2)
)

```

可以看到，`vsplit`、`hsplit`、`dsplit` 是通过传入不同的 `axis` 对 Tensor 进行拆分的。

## Numpy

`Numpy` 提供了 `array_split` 接口，对应 `PyTorch` 的 `tensor_split` 函数。另外，也提供了 `hsplit`、`dsplit` 接口。

相应文档：

- [numpy.array_split](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
- [numpy.hsplit](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html)
- [numpy.dsplit](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html)

相应 python 实现为：

- [shape_base.py](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/shape_base.py) : core/shape_base.py

相应接口为：

- `numpy.array_split(ary, indices_or_sections, axis=0)`

    - 文档描述
    > Split an array into multiple sub-arrays.

    - 参数列表
    > ary – array to split.
    > indices_or_sections (int or list or tuple of ints) 
    > axis

    - 返回值
    > output (List of arrays)

    - 源码
    ``` python
    def array_split(ary, indices_or_sections, axis=0):
        try:
            Ntotal = ary.shape[axis]
        except AttributeError:
            Ntotal = len(ary)
        try:
            # handle array case.
            Nsections = len(indices_or_sections) + 1
            div_points = [0] + list(indices_or_sections) + [Ntotal]
        except TypeError:
            # indices_or_sections is a scalar, not an array.
            Nsections = int(indices_or_sections)
            if Nsections <= 0:
                raise ValueError('number sections must be larger than 0.') from None
            Neach_section, extras = divmod(Ntotal, Nsections)
            section_sizes = ([0] +
                            extras * [Neach_section+1] +
                            (Nsections-extras) * [Neach_section])
            div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

        sub_arys = []
        sary = _nx.swapaxes(ary, axis, 0)
        for i in range(Nsections):
            st = div_points[i]
            end = div_points[i + 1]
            sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))

        return sub_arys
    ```

- `numpy.hsplit(ary, indices_or_sections)`

    - 文档描述
    > Split an array into multiple sub-arrays horizontally (column-wise).

    - 参数列表
    > ary – array to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of arrays)

    - 源码
    ``` python
    def hsplit(ary, indices_or_sections):
        if _nx.ndim(ary) == 0:
            raise ValueError('hsplit only works on arrays of 1 or more dimensions')
        if ary.ndim > 1:
            return split(ary, indices_or_sections, 1)
        else:
            return split(ary, indices_or_sections, 0)
    ```

- `numpy.dsplit(ary, indices_or_sections)`

    - 文档描述
    > Split array into multiple sub-arrays along the 3rd axis (depth).

    - 参数列表
    > ary – array to split.
    > indices_or_sections (int or list or tuple of ints) 

    - 返回值
    > output (List of arrays)

    - 源码
    ``` python
    def dsplit(ary, indices_or_sections):
        if _nx.ndim(ary) < 3:
            raise ValueError('dsplit only works on arrays of 3 or more dimensions')
        return split(ary, indices_or_sections, 2)
    ```

可以看到，`hsplit`、`dsplit` 可以通过 `split` 函数实现。

# 四、对比分析

`PyTorch`、和 `Numpy` 均提供了上层 python 接口，`PyTorch` 进一步调用底层的 c++ 函数。

`TensorFlow` 缺少 `tensor_split` 对应的接口。

`TensorFlow`、`Numpy` 的 `hsplit`、`dsplit` 均通过 `split` 实现，而 `PyTorch` 的相应接口通过 `tensor_split` 实现。

另一方面，`hsplit`、`dsplit`、`vsplit` 是一组功能类似的接口，`Paddle` 通过 `split` 接口实现了 `vsplit` 函数，因此，可以考虑与 `TensorFlow`、`Numpy` 相同的方式，使用 `split` 接口实现。

# 五、设计思路与实现方案

## 命名与参数设计

添加 python 上层接口:

- `paddle.tensor_split(x, num_or_sections, axis=0)`
- `Tensor.tensor_split(num_or_sections, axis=0)`

    - 参数列表
    > x (Tensor) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > num_or_sections (Tensor|int|list|tuple) 
    > axis (int, optional) – dimension along which to split the tensor. Default: 0

    - 返回值
    > output (List of Tensors)

- `paddle.hsplit(x, num_or_sections)`
- `Tensor.hsplit(num_or_sections)`

    - 参数列表
    > x (Tensor) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > num_or_sections (Tensor|int|list|tuple) 

    - 返回值
    > output (List of Tensors)

- `paddle.dsplit(x, num_or_sections)`
- `Tensor.dsplit(num_or_sections)`

    - 参数列表
    > x (Tensor) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > num_or_sections (Tensor|int|list|tuple) 

    - 返回值
    > output (List of Tensors)

另外，这几个接口均无需 `name` 参数，因为输出可能为多个 Tensor。

## 底层 OP 设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案

- 利用目前 `Paddle` 已有的 `split`、`slice` 等接口实现。
- 加入 `Paddle` 公共 API
- 将 API 绑定为 Tensor 的方法

具体接口：

- `paddle.tensor_split(x, num_or_sections, axis=0)`

    ``` python
    from paddle.base.framework import Variable
    def tensor_split(x, num_or_sections, axis=0):
        if x.ndim < 1:
            raise ValueError(
                f"The input tensor's dimension must be greater than 0, but got {x.ndim}"
            )

        total_n = paddle.full(shape=(), fill_value=x.shape[axis], dtype=x.dtype)

        def _tensor_split_int(total_n, sections, axis):
            section = paddle.divide(total_n, sections)

            splits = []
            starts = paddle.full(shape=(), fill_value=0, dtype=section.dtype)
            ends = paddle.add(section, paddle.full(shape=(), fill_value=1, dtype=section.dtype))

            while starts < total_n:
                sub_array = paddle.slice(x, axes=[axis], starts=[starts], ends=[ends])
                splits.append(sub_array)
                starts = ends
                ends = paddle.add(ends, paddle.add(section, paddle.full(shape=(), fill_value=1, dtype=section.dtype)))

            return splits

        def _tensor_split_array(total_n, sections, axis):
            splits = []

            starts = 0
            ends = sections[0]
            sub_array = paddle.slice(x, axes=[axis], starts=[starts], ends=[ends])
            splits.append(sub_array)

            for idx in sections[1:]:
                starts = ends
                ends = idx
                sub_array = paddle.slice(x, axes=[axis], starts=[starts], ends=[ends])
                splits.append(sub_array)

            starts = ends
            ends = total_n
            sub_array = paddle.slice(x, axes=[axis], starts=[starts], ends=[ends])
            splits.append(sub_array)

            return splits

        sections = 0
        indices = []
        if isinstance(num_or_sections, int):
            return _tensor_split_int(total_n, paddle.full(shape=(), fill_value=num_or_sections, dtype=x.dtype), axis)

        elif isinstance(num_or_sections, Variable):
            if num_or_sections.ndim == 0:
                return _tensor_split_int(total_n, num_or_sections, axis)

            elif num_or_sections.ndim == 1:
                return _tensor_split_array(total_n, num_or_sections, axis)

            else:
                raise ValueError(
                    f"The num_or_sections tensor's dimension must be less than 2, but got {num_or_sections.ndim}"
                )

        elif isinstance(num_or_sections, (list, tuple)):
            return _tensor_split_array(total_n, num_or_sections, axis)

        else:
            raise ValueError(
                f"The num_or_sections should be Tensor, int or list or tuple of ints, but got {type(num_or_sections)}"
            )
    ```

    **疑问**： 静态图目前调不通，`while starts < total_n` 似乎永远跳不出来，还请指教！

- `paddle.hsplit(x, num_or_sections)`

    ``` python

    def hsplit(x, num_or_sections, name=None):
        if x.ndim < 1:
            raise ValueError(
                f"The input tensor's dimension must be greater than 0, but got {x.ndim}"
            )
        return split(x, num_or_sections, axis=1)
    ```

- `paddle.dsplit(x, num_or_sections)`

    ``` python

    def dsplit(x, num_or_sections, name=None):
        if x.ndim < 3:
            raise ValueError(
                f"The input tensor's dimension must be greater than 2, but got {x.ndim}"
            )
        return split(x, num_or_sections, axis=2)
    ```

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试 `num_or_sections` 为 int/list/tuple
  - 需要测试 `num_or_sections` 为 0D Tensor，1D Tensor
  - 需要测试 `num_or_sections` 为错误的维度

- **计算精度**
  需要保证前向计算的精度正确性，通过 numpy 实现的函数的对比结果

- **维度测试**
  - Paddle API 支持的最低维度为 0 维，单测中应编写相应的 0 维尺寸测试 case

# 七、可行性分析及规划排期

- 每个接口开发约 1 个工作日
- 每个接口测试约 1 个工作日

计划 1 周的工作量可以完成接口的开发预测是。

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

- [Pytorch文档学习 TORCH.TENSOR_SPLIT](https://blog.csdn.net/Jamesgender/article/details/130559738)
- [TORCH.TENSOR_SPLIT](https://pytorch.org/docs/stable/generated/torch.tensor_split.html?highlight=tensor_split)
- [TORCH.HSPLIT](https://pytorch.org/docs/stable/generated/torch.hsplit.html?highlight=hsplit)
- [TORCH.DSPLIT](https://pytorch.org/docs/stable/generated/torch.dsplit.html?highlight=dsplit)
- [tf.experimental.numpy.hsplit](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/hsplit?hl=en)
- [tf.experimental.numpy.dsplit](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/dsplit?hl=en)

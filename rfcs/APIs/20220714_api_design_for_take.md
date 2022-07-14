# paddle.Tensor.take 设计文档

|API名称 | paddle.take | 
|---|---|
|提交作者 | S-HuaBomb | 
|提交时间 | 2022-07-14 | 
|版本号   | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20220714_api_design_for_take.md | 

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度， Paddle 需要支持 API `paddle.take` 的功能。

## 2、功能目标

增加 API `paddle.take`，对于输入的 Tensor，将输入 Tensor 视为一维 Tensor，实现根据索引返回指定索引上的元素集合组成的新 Tensor。返回结果与索引的形状相同。

## 3、意义

为 paddle 增加 Tensor 索引函数，丰富 paddle Tensor 的索引功能。

# 二、飞桨现状

目前 paddle 可由 `Tensor.flatten`、`Tensor.index_select` 和 `Tensor.reshape` 组合实现该 API 的功能。

其主要实现逻辑为：

1. 通过 `Tensor.flatten()` 将输入 x 和 index 展开成 1D Tensor。

2. 通过 `Tensor.index_select(index)` 按照 index 中的索引提取对应元素。

3. 通过 `Tensor.reshape(index.shape)` 将输出的 Tensor 形状转成 index 的形状。

# 三、业内方案调研

## Pytorch

Pytorch 中有 API `torch.take(input, index) → Tensor`。在 pytorch 中，介绍为：

```
Returns a new tensor with the elements of input at the given indices. The input tensor is treated as if it were viewed as a 1-D tensor. The result takes the same shape as the indices.
```

### 实现方法

Pytorch 在 [代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L699) 中定义了 `take` 方法，核心代码为：

```c++
Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // Type and device checks
  TORCH_CHECK(index.scalar_type() == ScalarType::Long, "take(): Expected a long tensor for index, but got ", index.scalar_type())
  TORCH_CHECK(self.scalar_type() == out.scalar_type(), "take(): self and out expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and out.dtype = ", out.scalar_type());
  TORCH_CHECK(self.device() == out.device() && self.device() == index.device(),
      "take(): self, index and out expected to be in the same device, but got self.device = ",
      self.device(), ", index.device = ", index.device(), ", and out.device = ", out.device());

  // index checks
  TORCH_CHECK_INDEX(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");

  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, self);

  // Do not iterate over self, we will compute the offsets manually
  // out is resized inside tensor_iterator
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_output(out)
    .add_input(index)
    .build();

  // Early return after out has been resized
  if (index.numel() == 0) {
    return out;
  }

  take_stub(iter.device_type(), iter, self);

  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
    auto out = at::empty(index.sizes(), self.options());
    at::native::take_out(self, index, out);
    return out;
}
```

在 [代码位置](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py#L4932) 中也定义了 `take` 方法：

```python
def take(g, self, index):
    self_flattened = sym_help._reshape_helper(g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
    out = index_select(g, self_flattened, 0, index)
    out = reshape_as(g, out, index)
    return out
```

整体逻辑为：

- 通过 `Tensor.flatten()` 将输入 x 和 index 展开成 1D Tensor。

- 通过 `Tensor.index_select(index)` 按照 index 中的索引提取对应元素。

- 通过 `Tensor.reshape(index.shape)` 将输出的 Tensor 转成 index 的形状。


## Numpy

### 实现方法

Numpy 在[代码位置](https://github.com/numpy/numpy/blob/main/numpy/core/fromnumeric.py#L94)已经有该 API 的实现：`numpy.take(a, indices, axis=None, out=None, mode='raise')`。

其中核心代码为：

```Python
@array_function_dispatch(_take_dispatcher)
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
        The source array.
    indices : array_like (Nj...)
        The indices of the values to extract.

        .. versionadded:: 1.8.0

        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if `mode='raise'`; use other modes for better performance.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray (Ni..., Nj..., Nk...)
        The returned array has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    ndarray.take : equivalent method
    take_along_axis : Take elements by matching the array and the index arrays

    Notes
    -----

    By eliminating the inner loop in the description above, and using `s_` to
    build simple slice objects, `take` can be expressed  in terms of applying
    fancy indexing to each 1-d slice::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nj):
                out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

    For this reason, it is equivalent to (but faster than) the following use
    of `apply_along_axis`::

        out = np.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

    Examples
    --------
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    In this example if `a` is an ndarray, "fancy" indexing can be used.

    >>> a = np.array(a)
    >>> a[indices]
    array([4, 3, 6])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])
    """
    return _wrapfunc(a, 'take', indices, axis=axis, out=out, mode=mode)
```
整体逻辑为：

- 当指定 `axis` 的时候，`numpy.take` 执行与 “fancy indexing” 相同的索引操作（使用数组索引数组）；例如 `np.take(arr, indices, axis=3)` 等价于 `arr[:, :, :, indices, ...]`。

- 当不指定 `axis` 的时候，`numpy.take` 默认将输入展平再使用 “fancy indexing”。

- 当提供参数 `out` 的时候，输出的数据将填充到 `out` 中。

# 四、对比分析

- `torch.take` 的 `index` 参数必须为 LongTensor 类型，`numpy.take` 的 `indices` 参数直接取整。

- 当不指定轴时，对于相同的索引矩阵，`Numpy.take` 的执行结果等于 `torch.take`。

- 在维度支持上，`Numpy.take` 支持指定轴，`torch.take` 不支持。

- `Numpy.take` 支持通过 `mode` 参数指定索引越界的 3 种处理方式，默认直接报错；`torch.take` 在索引越界时直接报错。

# 五、方案设计

## 命名与参数设计

添加 API

```python
paddle.take(
  input: Tensor,
  index: Tensor,
  name: str=None)
```

注：其中添加参数 `name` 为了与 Paddle 其他 API 参数名保持一致。

## 底层 OP 设计

使用已有 API 组合实现，不再单独设计 OP。

## API 实现方案

该 API 需要添加在 Paddle repo 的 `python/paddle/tensor/math.py` 文件中；并在 `python/paddle/tensor/init.py` 中添加 `take` API，以支持 Tensor.take 的调用方式。

目前 paddle 可由 `Tensor.flatten`、`Tensor.index_select` 和 `Tensor.reshape` 组合实现该 API 的功能。

其主要实现逻辑为：

1. 通过 `Tensor.flatten()` 将输入 x 和 index 展开成 1D Tensor。

2. 通过 `Tensor.index_select(index)` 按照 index 中的索引提取对应元素。

3. 通过 `Tensor.reshape(index.shape)` 将输出的 Tensor 形状转成 index 的形状。

# 六、测试和验收的考量

测试考虑的 case 如下：

- 参数 `index` 数据类型必须为 `paddle.int64` 类型（与 `torch.long` 同精度）。

- `index` 索引越界时直接报错。

- 在动态图、静态图下的都能得到正确的结果。

# 七、可行性分析及规划排期

可行性分析：

目前 paddle 可由 `Tensor.flatten`、`Tensor.index_select` 和 `Tensor.reshape` 组合实现该 API 的功能。

具体规划为：

- 阶段一：在 Paddle repo 的 `python/paddle/tensor/math.py` 文件中实现代码 & 英文 API 文档，；并在 `python/paddle/tensor/init.py` 中，添加 `take` API，以支持 Tensor.take 的调用方式。

- 阶段二：单测代码，在 Paddle repo 的 `python/paddle/fluid/tests/unittests` 目录添加测试代码。

- 阶段三：中文API文档，在 docs repo 的 `docs/api/paddle` 目录和 `docs/api/paddle/Tensor_cn.rst` 文件，同时需要在 `docs/api/paddle/Overview_cn.rst` 文件中添加 API 介绍。

# 八、影响面

增加了一个 `paddle.take` API。

# 名词解释

无

# 附件及参考资料

无

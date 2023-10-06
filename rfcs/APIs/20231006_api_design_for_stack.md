# column_stack / row_stack / dstack / hstack / vstack API 设计文档

| API 名称 | column_stack / row_stack / dstack / hstack / vstack |
| - | - |
| 提交作者 | megemini(柳顺) |
| 提交时间 | 2023-10-06 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20231006_api_design_for_stack.md |


# 一、概述

## 1、相关背景

在深度学习、机器学习、图像处理、矩阵运算等场景中，往往需要将多个数组按照轴向堆叠起来，如 `column_stack` 用于将多个数组按列堆叠（垂直堆叠）在一起。

目前 `Paddle` 框架中没有 `column_stack / row_stack / dstack / hstack / vstack` 函数，特在此任务中实现，以提升飞桨 API 的丰富程度。

## 2、功能目标

将一个 Tensor 根据不同方式堆叠拼接成一个 Tensor，根据不同的轴与操作方式，分别可以有多种 `stack`。该API依赖黑客松其他任务：`atleast_1d / atleast_2d / atleast_3d`，可在该任务完成之后再开发。不同API的调用路径为：

- `paddle.column_stack`，作为独立的函数调用
- `Tensor.column_stack`，作为 Tensor 的方法使用
- `paddle.row_stack`，作为独立的函数调用
- `Tensor.row_stack`，作为 Tensor 的方法使用
- `paddle.dstack`，作为独立的函数调用
- `Tensor.dstack`，作为 Tensor 的方法使用
- `paddle.hstack`，作为独立的函数调用
- `Tensor.hstack`，作为 Tensor 的方法使用
- `paddle.vstack`，作为独立的函数调用
- `Tensor.vstack`，作为 Tensor 的方法使用

**疑问** ： 这些接口为什么会有 `Tensor.XXX` 的方法？一般堆叠都是对多个 Tensor 的操作，另外，`PyTorch`、`Numpy` 也没有类似的使用方法，还请确认一下。 

## 3、意义

为 `Paddle` 增加 `column_stack / row_stack / dstack / hstack / vstack` 操作，丰富 `Paddle` 中张量视图的 API。

# 二、飞桨现状

目前 `Paddle` 在 python 端缺少相关接口的实现，而在底层也没有相关算子。

`python/paddle/tensor/manipulation.py` 文件中实现了若干对于 `Tensor` 操作的接口，如 `concat` 等。

另外，`atleast_1d`、`atleast_2d`、`atleast_3d` 同样在此次任务之前进行实现，可以利用这些接口实现本次目标。

# 三、业内方案调研

## PyTorch

`PyTorch` 底层通过 c++ 实现 `column_stack / row_stack / dstack / hstack / vstack` 函数，并通过上层的 python 对外开放相应接口。

相应文档：

- [TORCH.COLUMN_STACK](https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch-column-stack)
- [TORCH.ROW_STACK](https://pytorch.org/docs/stable/generated/torch.row_stack.html#torch-row-stack)
- [TORCH.DSTACK](https://pytorch.org/docs/stable/generated/torch.dstack.html#torch-dstack)
- [TORCH.HSTACK](https://pytorch.org/docs/stable/generated/torch.hstack.html#torch-hstack)
- [TORCH.VSTACK](https://pytorch.org/docs/stable/generated/torch.vstack.html#torch-vstack)

c++ 接口文件在：

- [TensorShape.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp) : aten/src/ATen/native/TensorShape.cpp

相应接口为：

- `torch.column_stack(tensors, *, out=None) → Tensor`

    - 文档描述
    > Creates a new tensor by horizontally stacking the tensors in tensors.

    - 参数列表
    > tensors (sequence of Tensors) – sequence of tensors to concatenate
    > out (Tensor, optional) – the output tensor.

    - 返回值
    > output (Tensor)

    - 源码
    ``` cpp
    static std::vector<Tensor> reshape_input_for_column_stack(TensorList tensors) {
    std::vector<Tensor> result(tensors.size());
    auto transform_lambda = [](const Tensor& input) -> Tensor {
        // reshape 0D or 1D tensor t into (t.numel(), 1)
        if (input.dim() <= 1) {
        return input.reshape_symint({input.sym_numel(), 1});
        }
        return input;
    };
    std::transform(tensors.cbegin(),
                    tensors.cend(),
                    result.begin(),
                    transform_lambda);
    return result;
    }

    Tensor& column_stack_out(TensorList tensors, Tensor& result) {
    TORCH_CHECK(!tensors.empty(),
                "column_stack expects a non-empty TensorList");

    auto reshaped_tensors = reshape_input_for_column_stack(tensors);
    return at::hstack_out(result, reshaped_tensors);
    }

    Tensor column_stack(TensorList tensors) {
    TORCH_CHECK(!tensors.empty(),
                "column_stack expects a non-empty TensorList");

    auto reshaped_tensors = reshape_input_for_column_stack(tensors);
    return at::hstack(reshaped_tensors);
    }
    ```

- `torch.row_stack(tensors, *, out=None) → Tensor`

    - 文档描述
    > Alias of torch.vstack().

    - 参数列表
    > tensors (sequence of Tensors) – sequence of tensors to concatenate
    > out (Tensor, optional) – the output tensor.

    - 返回值
    > output (Tensor)

    - 源码
    ``` cpp
    // torch.row_stack, alias for torch.vstack
    Tensor& row_stack_out(TensorList tensors, Tensor& result) {
    return at::vstack_out(result, tensors);
    }

    Tensor row_stack(TensorList tensors) {
    return at::vstack(tensors);
    }
    ```

- `torch.dstack(tensors, *, out=None) → Tensor`

    - 文档描述
    > Stack tensors in sequence depthwise (along third axis).

    - 参数列表
    > tensors (sequence of Tensors) – sequence of tensors to concatenate
    > out (Tensor, optional) – the output tensor.

    - 返回值
    > output (Tensor)

    - 源码
    ``` cpp
    Tensor dstack(TensorList tensors) {
    TORCH_CHECK(!tensors.empty(),
            "dstack expects a non-empty TensorList");
    auto rep = at::atleast_3d(tensors);
    return at::cat(rep, 2);
    }
    Tensor& dstack_out(TensorList tensors, Tensor& result) {
    TORCH_CHECK(!tensors.empty(),
            "dstack expects a non-empty TensorList");
    auto rep = at::atleast_3d(tensors);
    return at::cat_out(result, rep, 2);
    }
    ```

- `torch.hstack(tensors, *, out=None) → Tensor`

    - 文档描述
    > Stack tensors in sequence horizontally (column wise).

    - 参数列表
    > tensors (sequence of Tensors) – sequence of tensors to concatenate
    > out (Tensor, optional) – the output tensor.

    - 返回值
    > output (Tensor)

    - 源码
    ``` cpp
    Tensor hstack(TensorList tensors) {
    TORCH_CHECK(!tensors.empty(),
            "hstack expects a non-empty TensorList");
    auto rep = at::atleast_1d(tensors);
    if (rep[0].dim() == 1) {
        return at::cat(rep, 0);
    }
    return at::cat(rep, 1);
    }

    Tensor& hstack_out(TensorList tensors, Tensor& result) {
    TORCH_CHECK(!tensors.empty(),
            "hstack expects a non-empty TensorList");
    auto rep = at::atleast_1d(tensors);
    if (rep[0].dim() == 1) {
        return at::cat_out(result, rep, 0);
    }
    return at::cat_out(result, rep, 1);
    }
    ```

- `torch.vstack(tensors, *, out=None) → Tensor`

    - 文档描述
    > Stack tensors in sequence vertically (row wise).

    - 参数列表
    > tensors (sequence of Tensors) – sequence of tensors to concatenate
    > out (Tensor, optional) – the output tensor.

    - 返回值
    > output (Tensor)

    - 源码
    ``` cpp
    Tensor vstack(TensorList tensors) {
    TORCH_CHECK(!tensors.empty(),
            "vstack expects a non-empty TensorList");
    auto rep = at::atleast_2d(tensors);
    return at::cat(rep, 0);
    }

    Tensor& vstack_out(TensorList tensors, Tensor& result) {
    TORCH_CHECK(!tensors.empty(),
            "vstack expects a non-empty TensorList");
    auto rep = at::atleast_2d(tensors);
    return at::cat_out(result, rep, 0);
    }
    ```

可以看到，上面的几个函数都有很多相似的地方，比如 `row_stack` 就是 `Alias of torch.vstack().`。而具体实现也都可以通过 `cat` 操作完成。

## TensorFlow

`TensorFlow` 并没有 `column_stack / row_stack` 函数，只实现了 `dstack / hstack / vstack` 函数。

相应文档：

- [tf.experimental.numpy.dstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/dstack?hl=en)
- [tf.experimental.numpy.hstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/hstack?hl=en)
- [tf.experimental.numpy.vstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/vstack?hl=en)

`TensorFlow` 的 `dstack / hstack / vstack` 函数是一种 `Numpy` 实现的变体。

- [np_array_ops.py](https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/numpy_ops/np_array_ops.py) : python/ops/numpy_ops/np_array_ops.py

相应接口为：

- `tf.experimental.numpy.dstack`

    - 文档描述
    > TensorFlow variant of NumPy's dstack.

    - 参数列表
    > tup – sequence of tensors to concatenate

    - 返回值
    > output (Tensor)

    - 源码
    ``` python
    def dstack(tup):
        arrays = [atleast_3d(a) for a in tup]
        arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
        unwrapped_arrays = [
            a if isinstance(a, np_arrays.ndarray) else a for a in arrays
        ]
        return array_ops.concat(unwrapped_arrays, axis=2)
    ```

- `tf.experimental.numpy.hstack`

    - 文档描述
    > TensorFlow variant of NumPy's hstack.

    - 参数列表
    > tup – sequence of tensors to concatenate

    - 返回值
    > output (Tensor)

    - 源码
    ``` python
    def hstack(tup):
        arrays = [atleast_1d(a) for a in tup]
        arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
        unwrapped_arrays = [
            a if isinstance(a, np_arrays.ndarray) else a for a in arrays
        ]
        rank = array_ops.rank(unwrapped_arrays[0])
        return np_utils.cond(
            math_ops.equal(rank, 1),
            lambda: array_ops.concat(unwrapped_arrays, axis=0),
            lambda: array_ops.concat(unwrapped_arrays, axis=1),
        )
    ```

- `tf.experimental.numpy.vstack`

    - 文档描述
    > TensorFlow variant of NumPy's vstack.

    - 参数列表
    > tup – sequence of tensors to concatenate

    - 返回值
    > output (Tensor)

    - 源码
    ``` python
    def vstack(tup):
        arrays = [atleast_2d(a) for a in tup]
        arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
        unwrapped_arrays = [
            a if isinstance(a, np_arrays.ndarray) else a for a in arrays
        ]
        return array_ops.concat(unwrapped_arrays, axis=0)
    ```

## Numpy

`Numpy` 提供了 `column_stack / row_stack / dstack / hstack / vstack` 接口。

相应文档：

- [numpy.column_stack](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy-column-stack)
- [numpy.row_stack](https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html#numpy-row-stack)
- [numpy.dstack](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy-dstack)
- [numpy.hstack](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy-hstack)
- [numpy.vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack)

相应 python 实现为：

- [column_stack](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/shape_base.py) : lib/shape_base.py
- [row_stack](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/shape_base.py) : lib/shape_base.py
- [dstack](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/shape_base.py) : lib/shape_base.py
- [hstack](https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/shape_base.py) : core/shape_base.py
- [vstack](https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/shape_base.py) : core/shape_base.py

**注意**：这里实现分别在 `lib/shape_base.py` 与 `core/shape_base.py` 两个文件中。

相应接口为：

- `numpy.column_stack(tup)`

    - 文档描述
    > Stack 1-D arrays as columns into a 2-D array.

    - 参数列表
    > tup : equence of 1-D or 2-D arrays.

    - 返回值
    > stacked : 2-D array

    - 源码
    ``` python
    def column_stack(tup):
        arrays = []
        for v in tup:
            arr = asanyarray(v)
            if arr.ndim < 2:
                arr = array(arr, copy=False, subok=True, ndmin=2).T
            arrays.append(arr)
        return _nx.concatenate(arrays, 1)
    ```

- `numpy.row_stack(tup, *, dtype=None, casting='same_kind')`

    - 文档描述
    > Stack arrays in sequence vertically (row wise).

    - 参数列表
    > tup : sequence of ndarrays
    > dtype : str or dtype
    > casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional

    - 返回值
    > stacked: nd array

    - 源码
    ``` python
    row_stack = vstack
    ```

- `numpy.dstack(tup)`

    - 文档描述
    > Stack arrays in sequence depth wise (along third axis).

    - 参数列表
    > tup : sequence of ndarrays

    - 返回值
    > stacked: nd array

    - 源码
    ``` python
    def dstack(tup):
        arrs = atleast_3d(*tup)
        if not isinstance(arrs, list):
            arrs = [arrs]
        return _nx.concatenate(arrs, 2)
    ```

- `numpy.hstack(tup, *, dtype=None, casting='same_kind')`

    - 文档描述
    > Stack arrays in sequence horizontally (column wise).

    - 参数列表
    > tup : sequence of ndarrays
    > dtype : str or dtype
    > casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional

    - 返回值
    > stacked: nd array

    - 源码
    ``` python
    def hstack(tup, *, dtype=None, casting="same_kind"):
        arrs = atleast_1d(*tup)
        if not isinstance(arrs, list):
            arrs = [arrs]
        # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
        if arrs and arrs[0].ndim == 1:
            return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
        else:
            return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)
    ```

- `numpy.vstack(tup, *, dtype=None, casting='same_kind')`

    - 文档描述
    > Stack arrays in sequence vertically (row wise).

    - 参数列表
    > tup : sequence of ndarrays
    > dtype : str or dtype
    > casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional

    - 返回值
    > stacked: nd array

    - 源码
    ``` python
    def vstack(tup, *, dtype=None, casting="same_kind"):
        arrs = atleast_2d(*tup)
        if not isinstance(arrs, list):
            arrs = [arrs]
        return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)

    ```

# 四、对比分析

`PyTorch`、和 `Numpy` 均提供了上层 python 接口，`PyTorch` 进一步调用底层的 c++ 函数。

`TensorFlow` 缺少 `column_stack / row_stack` 对应的接口。

另外，三者在实现逻辑上基本一样。

# 五、设计思路与实现方案

## 命名与参数设计

添加 python 上层接口:

- `paddle.column_stack(x, name=None)`
- `Tensor.column_stack` (有疑问)

    - 参数列表
    > x (List of Tensors) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > name (str, optional): Name for the operation (optional, default is None). 

    - 返回值
    > output (Tensor)

- `paddle.row_stack(x, name=None)`
- `Tensor.row_stack` (有疑问)

    - 参数列表
    > x (List of Tensors) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > name (str, optional): Name for the operation (optional, default is None). 

    - 返回值
    > output (Tensor)

- `paddle.dstack(x, name=None)`
- `Tensor.dstack`  (有疑问)

    - 参数列表
    > x (List of Tensors) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > name (str, optional): Name for the operation (optional, default is None). 

    - 返回值
    > output (Tensor)

- `paddle.hstack(x, name=None)`
- `Tensor.hstack`  (有疑问)

    - 参数列表
    > x (List of Tensors) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > name (str, optional): Name for the operation (optional, default is None). 

    - 返回值
    > output (Tensor)

- `paddle.vstack(x, name=None)`
- `Tensor.vstack`  (有疑问)

    - 参数列表
    > x (List of Tensors) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > name (str, optional): Name for the operation (optional, default is None). 

    - 返回值
    > output (Tensor)

## 底层 OP 设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案

- 利用目前 `Paddle` 已有的 `concate`、`atleast_1d`、`atleast_2d`、`atleast_3d` 等接口实现。
- 加入 `Paddle` 公共 API
- 将 API 绑定为 Tensor 的方法 (有疑问)

具体接口：

- `paddle.column_stack(x, name=None)`

    ``` python
    def column_stack(x, name=None):
        arrays = []

        for tensor in x:
            if tensor.ndim < 2:
                arrays.append(tensor.reshape((tensor.numel(), 1)))
            else:
                arrays.append(tensor)

        return paddle.hstack(arrays, name=name)
    ```

- `paddle.row_stack(x, name=None)`

    ``` python
    def row_stack(x, name=None):
        return paddle.vstack(x, name=name)
    ```

- `paddle.dstack(x, name=None)`

    ``` python
    def dstack(x, name=None):
        arrays = paddle.atleast_3d(*x)
        if not isinstance(arrays, list):
            arrays = [arrays]

        return paddle.concat(arrays, axis=2, name=name)
    ```

- `paddle.hstack(x, name=None)`

    ``` python
    def hstack(x, name=None):
        arrays = paddle.atleast_1d(*x)
        if not isinstance(arrays, list):
            arrays = [arrays]

        if arrays and arrays[0].ndim == 1:
            return paddle.concat(arrays, axis=0, name=name)
        else:
            return paddle.concat(arrays, axis=1, name=name)
    ```

- `paddle.vstack(x, name=None)`

    ``` python
    def vstack(x, name=None):
        arrays = paddle.atleast_2d(*x)
        if not isinstance(arrays, list):
            arrays = [arrays]

        return paddle.concat(arrays, axis=0, name=name)
    ```

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试多个向量输入的方式
  - 需要测试向量维度不同的方式

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

- [TORCH.COLUMN_STACK](https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch-column-stack)
- [TORCH.ROW_STACK](https://pytorch.org/docs/stable/generated/torch.row_stack.html#torch-row-stack)
- [TORCH.DSTACK](https://pytorch.org/docs/stable/generated/torch.dstack.html#torch-dstack)
- [TORCH.HSTACK](https://pytorch.org/docs/stable/generated/torch.hstack.html#torch-hstack)
- [TORCH.VSTACK](https://pytorch.org/docs/stable/generated/torch.vstack.html#torch-vstack)
- [tf.experimental.numpy.dstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/dstack?hl=en)
- [tf.experimental.numpy.hstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/hstack?hl=en)
- [tf.experimental.numpy.vstack](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/vstack?hl=en)
- [numpy.column_stack](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy-column-stack)
- [numpy.row_stack](https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html#numpy-row-stack)
- [numpy.dstack](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy-dstack)
- [numpy.hstack](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy-hstack)
- [numpy.vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack)

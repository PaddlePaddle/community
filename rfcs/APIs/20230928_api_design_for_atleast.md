# paddle.atleast_1d/paddle.atleast_2d/paddle.atleast_3d 设计文档

| API 名称 | paddle.paddle.atleast_1d/paddle.atleast_2d/paddle.atleast_3d |
| - | - |
| 提交作者 | megemini(柳顺) |
| 提交时间 | 2023-09-28 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20230928_api_design_for_atleast.md |

# 一、概述

## 1、相关背景

`atleast_Nd` 接口包括 `atleast_1d`, `atleast_2d`, `atleast_3d` 这几个方法，目的是返回至少 `N` 维的张量，可以用于张量之间的对齐等操作。

目前的 `Paddle` 框架中暂无相关 API，特在此任务中实现，涉及接口为：

- `paddle.atleast_1d`
- `paddle.atleast_2d`
- `paddle.atleast_3d`

以提升飞桨 API 的丰富程度。

## 2、功能目标

- `paddle.atleast_1d` 作为独立的函数调用，返回每个零维输入张量的一维视图，有一个或多个维度的输入张量将按原样返回。
- `paddle.atleast_2d` 作为独立的函数调用，返回每个零维输入张量的二维视图，有两个或多个维度的输入张量将按原样返回。
- `paddle.atleast_3d` 作为独立的函数调用，返回每个零维输入张量的三维视图，有三个或多个维度的输入张量将按原样返回。

## 3、意义

为 `Paddle` 增加 `atleast_Nd` 操作，丰富 `Paddle` 中张量视图的 API。

# 二、飞桨现状

目前 `Paddle` 在 python 端缺少相关接口的实现，而在底层也没有相关算子。

`python/paddle/tensor/manipulation.py` 文件中实现了若干对于 `Tensor` 操作的接口，如 `reshape`, `unsqueeze` 等，可以利用已有的这些接口构造 `atleast_Nd` 方法。

# 三、业内方案调研

## PyTorch

`PyTorch` 底层提供 `atleast_Nd` 函数，并通过上层的 python 对外开放相应接口。`PyTorch` 对相应接口归类为 `Other Operations` 部分。

### Python 接口

其中，python 接口文件在：

- [functional](https://github.com/pytorch/pytorch/blob/main/torch/functional.py) : torch/functional.py

相应接口为：

- `torch.atleast_1d(*tensors)`

    - 文档描述
    > Returns a 1-dimensional view of each input tensor with zero dimensions. Input tensors with one or more dimensions are returned as-is.

    - 参数列表
    > input (Tensor or list of Tensors)

    - 返回值
    > output (Tensor or tuple of Tensors)

    - 源码
    ``` python
    def atleast_1d(*tensors):
        # This wrapper exists to support variadic args.
        if has_torch_function(tensors):
            return handle_torch_function(atleast_1d, tensors, *tensors)
        if len(tensors) == 1:
            tensors = tensors[0]
        return _VF.atleast_1d(tensors)  # type: ignore[attr-defined]
    
    ```

    > 注：此处省略了函数中的 docstring，以方便文档的阅读，后面相同处理。

- `torch.atleast_2d(*tensors)`

    - 文档描述
    > Returns a 2-dimensional view of each input tensor with zero dimensions. Input tensors with two or more dimensions are returned as-is.

    - 参数列表
    > input (Tensor or list of Tensors)

    - 返回值
    > output (Tensor or tuple of Tensors)

    - 源码
    ``` python
    def atleast_2d(*tensors):
        # This wrapper exists to support variadic args.
        if has_torch_function(tensors):
            return handle_torch_function(atleast_2d, tensors, *tensors)
        if len(tensors) == 1:
            tensors = tensors[0]
        return _VF.atleast_2d(tensors)  # type: ignore[attr-defined]

    ```

- `torch.atleast_3d(*tensors)`

    - 文档描述
    > Returns a 3-dimensional view of each input tensor with zero dimensions. Input tensors with three or more dimensions are returned as-is.

    - 参数列表
    > input (Tensor or list of Tensors)

    - 返回值
    > output (Tensor or tuple of Tensors)

    - 源码
    ``` python

    def atleast_3d(*tensors):
        # This wrapper exists to support variadic args.
        if has_torch_function(tensors):
            return handle_torch_function(atleast_3d, tensors, *tensors)
        if len(tensors) == 1:
            tensors = tensors[0]
        return _VF.atleast_3d(tensors)  # type: ignore[attr-defined]

    ```

分析这三个接口，实现方法类似，都是直接将 tensor 转发至底层的 c++ 对应的函数处理。

### c++ 接口

其中，c++ 接口文件在：

- [TensorTransformations](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorTransformations.cpp) : aten/src/ATen/native/TensorTransformations.cpp

相应接口为：

- `atleast_1d`

  ``` cpp
  Tensor atleast_1d(const Tensor& self) {
    switch (self.dim()) {
      case 0:
        return self.reshape({1});
      default:
        return self;
    }
  }

  std::vector<Tensor> atleast_1d(TensorList tensors) {
    std::vector<Tensor> result(tensors.size());
    auto transform_lambda = [](const Tensor& input) -> Tensor {
      return at::native::atleast_1d(input);
    };
    std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
    return result;
  }
  ```

- `atleast_2d`

  ``` cpp
  Tensor atleast_2d(const Tensor& self) {
    switch (self.dim()) {
      case 0:
        return self.reshape({1, 1});
      case 1: {
        return self.unsqueeze(0);
      }
      default:
        return self;
    }
  }

  std::vector<Tensor> atleast_2d(TensorList tensors) {
    std::vector<Tensor> result(tensors.size());
    auto transform_lambda = [](const Tensor& input) -> Tensor {
      return at::native::atleast_2d(input);
    };
    std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
    return result;
  }  
  ```

- `atleast_3d`

  ``` cpp
  Tensor atleast_3d(const Tensor& self) {
    switch (self.dim()) {
      case 0:
        return self.reshape({1, 1, 1});
      case 1: {
        return self.unsqueeze(0).unsqueeze(-1);
      }
      case 2: {
        return self.unsqueeze(-1);
      }
      default:
        return self;
    }
  }

  std::vector<Tensor> atleast_3d(TensorList tensors) {
    std::vector<Tensor> result(tensors.size());
    auto transform_lambda = [](const Tensor& input) -> Tensor {
      return at::native::atleast_3d(input);
    };
    std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
    return result;
  }
  ```

分析这三个接口，其中 `Tensor atleast_Nd(const Tensor& self)` 处理单个 tensor 的情况，`std::vector<Tensor> atleast_Nd(TensorList tensors)` 处理多个 tensor 的情况。

针对单个 tensor 的处理逻辑，都是判断 tensor 的 `dim` 后，通过 `reshape`、`unsqueeze` 进行转换。

## TensorFlow

`TensorFlow` 的 `atleast_Nd` 是一种 `Numpy` 实现的变体。

- [np_array_ops](https://github.com/tensorflow/tensorflow/blob/v2.14.0/tensorflow/python/ops/numpy_ops/np_array_ops.py) : python/ops/numpy_ops/np_array_ops.py

相应接口为：

- `tf.experimental.numpy.atleast_1d(*arys)`

    - 文档描述
    > TensorFlow variant of NumPy's atleast_1d.

    - 源码
    ``` python
    def atleast_1d(*arys):
      return _atleast_nd(1, _pad_left_to, *arys)
    ```

    其中 `_atleast_nd`:

    ``` python
    def _atleast_nd(n, new_shape, *arys):
      def f(x):
        # pylint: disable=g-long-lambda
        x = asarray(x)
        return asarray(
            np_utils.cond(
                np_utils.greater(n, array_ops.rank(x)),
                lambda: reshape(x, new_shape(n, array_ops.shape(x))),
                lambda: x,
            )
        )

      arys = list(map(f, arys))
      if len(arys) == 1:
        return arys[0]
      else:
        return arys
    ```

- `tf.experimental.numpy.atleast_2d(*arys)`

    - 文档描述
    > TensorFlow variant of NumPy's atleast_2d.

    - 源码
    ``` python
    def atleast_2d(*arys):
      return _atleast_nd(2, _pad_left_to, *arys)
    ```

- `tf.experimental.numpy.atleast_3d(*arys)`

    - 文档描述
    > TensorFlow variant of NumPy's atleast_3d.

    - 源码
    ``` python
    def atleast_3d(*arys):  # pylint: disable=missing-docstring
      def new_shape(_, old_shape):
        # pylint: disable=g-long-lambda
        ndim_ = array_ops.size(old_shape)
        return np_utils.cond(
            math_ops.equal(ndim_, 0),
            lambda: constant_op.constant([1, 1, 1], dtype=dtypes.int32),
            lambda: np_utils.cond(
                math_ops.equal(ndim_, 1),
                lambda: array_ops.pad(old_shape, [[1, 1]], constant_values=1),
                lambda: array_ops.pad(old_shape, [[0, 1]], constant_values=1),
            ),
        )

      return _atleast_nd(3, new_shape, *arys)
    ```

## Numpy

`Numpy` 提供了 `atleast_Nd` 接口，通过 python 直接实现。

- [shape_base](https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/shape_base.py) : core/shape_base.py

相应接口为：

- `numpy.atleast_1d(*arys)`

    - 文档描述
    > Convert inputs to arrays with at least one dimension. 
    Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.

    - 参数列表
    > arys1, arys2, … : array_like. One or more input arrays.

    - 返回值
    > ret : ndarray. An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.

    - 源码

    ``` python
    def atleast_1d(*arys):
        res = []
        for ary in arys:
            ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1)
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res    
    ```

- `numpy.atleast_2d(*arys)`

    - 文档描述
    > View inputs as arrays with at least two dimensions.

    - 参数列表
    > arys1, arys2, … : array_like.

    - 返回值
    > res, res2, … : ndarray

    - 源码

    ``` python
    def atleast_2d(*arys):
        res = []
        for ary in arys:
            ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1)
            elif ary.ndim == 1:
                result = ary[_nx.newaxis, :]
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res
    ```

- `numpy.atleast_3d(*arys)`

    - 文档描述
    > View inputs as arrays with at least three dimensions.

    - 参数列表
    > arys1, arys2, … : array_like.

    - 返回值
    > res, res2, … : ndarray

    - 源码

    ``` python
    def atleast_3d(*arys):
        res = []
        for ary in arys:
            ary = asanyarray(ary)
            if ary.ndim == 0:
                result = ary.reshape(1, 1, 1)
            elif ary.ndim == 1:
                result = ary[_nx.newaxis, :, _nx.newaxis]
            elif ary.ndim == 2:
                result = ary[:, :, _nx.newaxis]
            else:
                result = ary
            res.append(result)
        if len(res) == 1:
            return res[0]
        else:
            return res
    ```

# 四、对比分析

`PyTorch`、`TensorFlow` 和 `Numpy` 均提供了上层 python 接口，`PyTorch` 进一步调用底层的 c++ 函数。

`PyTorch`、`TensorFlow` 和 `Numpy` 对于函数入参，均提供了 `array like` 的方式，即，不仅仅支持单个 tensor 的调用，也支持 tensor 列表。

另外，此接口不涉及反向传播。

**注意** ： `list/tuple` 输入的处理方式，`PyTorch` 与 `Numpy` 不相同。

`PyTorch` 的处理方式：

``` python

In [41]: x = torch.tensor(0.3)
In [42]: y = torch.tensor(1.)

In [43]: torch.atleast_1d(x, y)
Out[43]: (tensor([0.3000]), tensor([1.]))

In [44]: torch.atleast_1d((x, y))
Out[44]: (tensor([0.3000]), tensor([1.]))

```

`Numpy` 的处理方式：

``` python

In [45]: a = np.array(0.3)
In [46]: b = np.array(1.)

In [47]: np.atleast_1d(a, b)
Out[47]: [array([0.3]), array([1.])]

In [48]: np.atleast_1d((a, b))
Out[48]: array([0.3, 1. ])

```

可以发现，`PyTorch` 对于 `atleast_Nd(x, y)` 这种的输入方式与 `atleast_Nd((x, y))` 处理结果一样，而 `Numpy` 将 `atleast_Nd(x, y)` 作为两个单独的向量处理，而  `atleast_Nd((x, y))` 作为一个向量处理。

这是两者对于 `多` 输入的处理方式的不同。

本文对于类似的输入，采用 `Numpy` 的处理方式，因为 `PyTorch` 的处理方式会对用户产生歧义。既然输入是 `*args` 的方式，则 `args` 中的每个元素都应该视作为一个整体进行处理，而 `PyTorch` 将其拆分开处理，不符合 python 对于 `*args` 方式的理解。如果需要拆分处理，可以采用 `atleast_Nd(x, y)` 方式的输入。


# 五、设计思路与实现方案

## 命名与参数设计

添加 python 上层接口:

- `paddle.atleast_1d(*inputs)`
- `paddle.atleast_2d(*inputs)`
- `paddle.atleast_3d(*inputs)`

参数：

- inputs: (Tensor|list(Tensor)) - 输入的一至多个 Tensor。数据类型支持：float32、float64、int32、int64。

返回：

Tensor 或 Tensor 列表。

## 底层 OP 设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案

利用目前 `Paddle` 已有的 `reshape`、`unsqueeze` 实现 `atleast_Nd` 接口。

- `paddle.atleast_1d(*inputs)`

    ``` python
    def atleast_1d(*inputs):
        res = []
        for tensor in inputs:
            tensor = paddle.to_tensor(tensor)
            if tensor.dim() == 0:
                result = tensor.reshape((1,))
            else:
                result = tensor
            res.append(result)

        if len(res) == 1:
            return res[0]
        else:
            return res    
    ```

- `paddle.atleast_2d(*inputs)`

    ``` python
    def atleast_2d(*inputs):
        res = []
        for tensor in inputs:
            tensor = paddle.to_tensor(tensor)
            if tensor.dim() == 0:
                result = tensor.reshape((1, 1))
            elif tensor.dim() == 1:
                result = paddle.unsqueeze(tensor, axis=0)
            else:
                result = tensor
            res.append(result)

        if len(res) == 1:
            return res[0]
        else:
            return res    
    ```

- `paddle.atleast_3d(*inputs)`

    ``` python
    def atleast_3d(*inputs):
        res = []
        for tensor in inputs:
            tensor = paddle.to_tensor(tensor)
            if tensor.dim() == 0:
                result = tensor.reshape((1, 1, 1))
            elif tensor.dim() == 1:
                result = paddle.unsqueeze(tensor, axis=[0, 2])
            elif tensor.dim() == 2:
                result = paddle.unsqueeze(tensor, axis=2)
            else:
                result = tensor
            res.append(result)

        if len(res) == 1:
            return res[0]
        else:
            return res    
    ```

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试单个向量、多个向量、`(向量 ... 向量)`，等方式
  - 需要测试数字与向量混合的方式

- **计算精度**
  需要保证前向计算的精度正确性，通过 numpy 实现的函数的对比结果

- **维度测试**
  - Paddle API 支持的最低维度为 0 维，单测中应编写相应的 0 维尺寸测试 case
  - 测试从 0 维至多维（`atleast_Nd` 中大于N）

# 七、可行性分析及规划排期

- 每个接口开发约 1 个工作日
- 每个接口测试约 1 个工作日

计划 1 周的工作量可以完成接口的开发预测是。

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

- [PyTorch atleast_1d 接口](https://pytorch.org/docs/stable/generated/torch.atleast_1d.html#torch.atleast_1d)
- [PyTorch atleast_2d 接口](https://pytorch.org/docs/stable/generated/torch.atleast_1d.html#torch.atleast_2d)
- [PyTorch atleast_3d 接口](https://pytorch.org/docs/stable/generated/torch.atleast_1d.html#torch.atleast_3d)
- [TensorFlow atleast_1d 接口](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/atleast_1d?hl=en)
- [TensorFlow atleast_2d 接口](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/atleast_2d?hl=en)
- [TensorFlow atleast_3d 接口](https://tensorflow.google.cn/api_docs/python/tf/experimental/numpy/atleast_3d?hl=en)
- [Numpy atleast_1d 接口](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html)
- [Numpy atleast_2d 接口](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html)
- [Numpy atleast_3d 接口](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html)

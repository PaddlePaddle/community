# paddle.unflatten 设计文档

| API 名称     | paddle.unflatten`/`paddle.nn.Unflatten |
| ------------ | -------------------------------------- |
| 提交作者     | [cos43](https://github.com/cos43)      |
| 提交时间     | 2023-03-14                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本 | develop                                |
| 文件名       | 20230314_api_design_for_unflatten.md   |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.unflatten，Tensor.unflatten，paddle.nn.Unflatten。

## 2、功能目标

实现 `unflatten` API，将输入 Tensor 的某一个维度，扩展成多个维度。

## 3、意义

为 paddle 框架中提供一种将输入 Tensor 的某一个维度，扩展成多个维度的 API。

# 二、飞桨现状

飞桨中提供了多种形状变换的 API，包括但不限于：

`paddle.reshape`: 改变 Tensor 的形状，可以将 Tensor 展平成一维或将一维 Tensor 展开成多维。

`paddle.flatten`: 将 Tensor 展平成一维。

目前，飞桨暂时没有直接将 Tensor 某一个维度展开成多个维度的 API。如果需要展开某一维度，可以先计算展开后的 shape，然后使用 `paddle.reshape` 展开。

# 三、业内方案调研

## Pytorch

Pytorch 中相关的 API 如下：

`torch.unflatten(input, dim, sizes)`

支持在多个维度上展开输入张量的维度。

Parameters:

- **input** ([_Tensor_](https://pytorch.org/docs/2.0/tensors.html#torch.Tensor)) – the input tensor.
- **dim** ([_int_](https://docs.python.org/3/library/functions.html#int)) – Dimension to be unflattened, specified as an index into `input.shape`.
- **sizes** (_Tuple_ _[_[_int_](https://docs.python.org/3/library/functions.html#int)_]_) – New shape of the unflattened dimension. One of its elements can be -1 in which case the corresponding output dimension is inferred. Otherwise, the product of `sizes` _must_ equal `input.shape[dim]`.

Returns:

A View of input with the specified dimension unflattened.

官方文档链接为：[torch.unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.unflatten.html?highlight=unflatten#torch.unflatten)

`torch.nn.Unflatten(dim, unflattened_size)`

展开一个张量，将其扩展到所需的形状。用于 Sequential。

Parameters:

- **dim** (_Union_ _[_[_int_](https://docs.python.org/3/library/functions.html#int)_,_ [_str_](https://docs.python.org/3/library/stdtypes.html#str)_]_) – Dimension to be unflattened
- **unflattened_size** (_Union**[**torch.Size,_ _Tuple_, _List_]) – New shape of the unflattened dimension

Shape:

- Input: (∗, $S_{dim}$ ,∗) where $S_{dim}$ is the size at dimension `dim` and ∗ means any number of dimensions including none.
- Output: (∗, $U_{1}$ ,..., $U_{n}$ ,∗), where $U$ = `unflattened_size` and $\prod_{i=1}^{n} U_i=S_{dim}$

官方文档链接：[Unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.nn.Unflatten.html?highlight=unflatten#torch.nn.Unflatten)

`Tensor.unflatten(dim, sizes)`

同`torch.unflatten(input, dim, sizes)`

官方文档链接：[torch.Tensor.unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.Tensor.unflatten.html?highlight=unflatten#torch.Tensor.unflatten)

#### Tensorflow

Tensorflow 没有直接 api，但是可以使用 reshape 达到相同的效果。

#### Numpy

Numpy 没有直接 api，但是可以使用 reshape 达到相同的效果。

# 实现方法

## Pytorch

PyTorch 中实现 unflatten 使用 C++ 代码实现，[实现代码](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp#L3418) 如下：

```c++
Tensor unflatten_impl(const Tensor& self, int64_t dim, SymIntArrayRef sizes, c10::optional<DimnameList> names) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!sizes.empty(), "unflatten: sizes must be non-empty");
  TORCH_INTERNAL_ASSERT(!names || names->size() == sizes.size());
  if (self.has_names()) {
    TORCH_CHECK(names, "unflatten: input is a named tensor but no names were given for unflattened sizes");
  }

  SymDimVector inferred_size;
  try {
    inferred_size = at::infer_size_dv(sizes, self.sym_size(dim));
  } catch (const std::runtime_error& e) {
    // at::infer_size would throw std::runtime_error for invalid size,
    // catch the runtime_error and display the error message in a more user-friendly way
    // for both tensors and named tensors
    handle_unflatten_exception(e, self, dim, sizes, names);
  }

  SymDimVector shape(self.sym_sizes().begin(), self.sym_sizes().end());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, inferred_size.begin(), inferred_size.end());

  Tensor result;
  {
    NoNamesGuard guard;
    result = self.view_symint(shape);
  }

  if (names) {
    auto outnames = self.names().vec();
    outnames.erase(outnames.begin() + dim);
    outnames.insert(outnames.begin() + dim, names->begin(), names->end());
    at::internal_set_names_inplace(result, outnames);
  }

  return result;
}

```

# 四、对比分析

计算思路基本一致，使用`paddle.reshape`完成。

paddle.unflatten API 的设计主要参考 PyTorch 中的实现，PyTorch 中`unflatten`具体逻辑如下：

- 验证输入参数
- 根据输入的参数，计算展开后的张量的形状
- 使用 torch.view_symint 方法将输入张量调整为展开后的形状
- 返回输出张量

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.unflatten(x, axis, shape, name=None)`

参数说明如下：

- **x** (Tensor) – 要进行扩展的张量，数据类型支持`uint8` `int8` `int16` `int32` `int64` `float32` `float64` `float16` `bfloat16` `complex64` `complex128` `bool` 数据类型。
- **axis** (int) – 要展开维度的轴，作为 `x.shape` 的索引。
- **shape** (list|tuple|Tensor) - 在指定轴上将该维度展成 `shape`， 其中 `shape` 最多包含一个 -1，如果输入 `shape` 不包含 -1 ，则 `shape` 内元素累乘的结果应该等于 `x.shape[axis]`。数据类型为 `int`。如果 `shape` 的类型是 `list` 或 `tuple`，它的元素可以是整数或者形状为[]的 `Tensor`。如果 `shape` 的类型是 `Tensor`，则是 1-D 的 `Tensor`。
- **name** (str,optional) – 操作的名称，更多信息请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)。

返回：在 axis 维度扩展成 shape 形状的 tensor

`Tensor.unflatten(axis, shape, name=None)`

参数说明如下：

- **axis** (int) – 要展开维度的轴，作为 `x.shape` 的索引。
- **shape** (list|tuple|Tensor) - 在指定轴上将该维度展成 `shape`， 其中 `shape` 最多包含一个 -1，如果输入 `shape` 不包含 -1 ，则 `shape` 内元素累乘的结果应该等于 `x.shape[axis]`。数据类型为 `int`。如果 `shape` 的类型是 `list` 或 `tuple`，它的元素可以是整数或者形状为[]的 `Tensor`。如果 `shape` 的类型是 `Tensor`，则是 1-D 的 `Tensor`。
- **name** (str,optional) – 操作的名称，更多信息请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)。

返回：在 axis 维度扩展成 shape 形状的 tensor

`paddle.nn.Unflatten(axis, shape, name=None)`

参数说明如下：

- **axis** (int) – 要展开维度的轴，作为 `x.shape` 的索引。
- **shape** (list|tuple|Tensor) - 在指定轴上将该维度展成 `shape`， 其中 `shape` 最多包含一个 -1，如果输入 `shape` 不包含 -1 ，则 `shape` 内元素累乘的结果应该等于 `x.shape[axis]`。数据类型为 `int`。如果 `shape` 的类型是 `list` 或 `tuple`，它的元素可以是整数或者形状为[]的 `Tensor`。如果 `shape` 的类型是 `Tensor`，则是 1-D 的 `Tensor`。
- **name** (str,optional) – 操作的名称，更多信息请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)。

返回：`Layer object`

## 底层 OP 设计

使用 paddle 现有 API 进行设计，不涉及底层 OP 新增与开发。

## API 实现方案

这个方法的目的是将一个张量（tensor）在指定的轴（axis）上展开为指定的形状（shape）

- 检查输入参数是否合法，如果不合法，抛出相应的异常

- 计算扩展之后的 shape，计算方法为:

  - 如果 `shape` 为 `list` 或者 `tuple` 则：

  ```Python
  new_shape = (list(x.shape[:axis]) + list(shape) + list(x.shape[axis + 1 :]))
  ```

  - 如果 `shape` 为 `Tensor`，则：
    ```Python
    new_shape = paddle.concat(
            [
                paddle.shape(x)[:axis],
                paddle.cast(shape, 'int32'),
                paddle.shape(x)[axis + 1 :],
            ]
        )
    ```

- 最后，这个方法使用 paddle.reshape 函数将输入张量转换为新的形状，并返回结果

# 六、测试和验收的考量

1. 结果正确性:

   - 这个函数的测试和验收的目标是确保它能正确地将输入张量在指定轴上展开为指定形状，并且能处理各种异常情况。
   - 前向计算: `paddle.unflatten`计算结果与基于 `numpy` 实现 `unflatten` 计算结果一致，其中基于`numpy` 实现 `unflatten` 在本地与 `torch.unflatten` 计算结果对齐。

   - 反向计算:由 Python 组合新增 API 无需验证反向计算。

2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。
3. 异常测试：
   - 类型检查
     - x 要求为 `paddle.Tensor` 类型。
       - x 为`paddle.Tensor` 类型
     - shape 要求为 `tuple`|`list`|`paddle.Tensor `类型。
       - shape 为 `list` 类型，例如：`[1,2,3]`
       - shape 为 `tuple` 类型，例如：`(1,2,3)`
       - shape 为 `paddle.Tensor` 类型，例如: `paddle.to_tensor([1, 2])`
     - axis 要求为 `int` 类型。
       - axis 为 int 类型，例如 `2`
     - name 若有输入，要求为 `str` 类型。
   - 数据类型检查
     - x 数据类型要求为 `uint8` `int8` `int16` `int32` `int64` `float32` `float64` `float16` `bfloat16` `complex64` `complex128` `bool`（几乎都支持）
     - shape
       - 如果`shape`是`list` 或者 `tuple`, 其元素必须为`int`类型。
         - shape 为`[1,2,3]`
         - shape 为`(1,2,3)`
       - 如果`shape`是`Tensor`，那么`shape`必须是一个维度为 1-D 的`Tensor`，数据类型应为`int32`。
         - shape 的类型不能为`float32` `float64` `float16` `bfloat16` `complex64` `complex128` `bool`
         - `paddle.to_tensor([1,2,3],dtype="bool")`(异常)
   - 具体数值检查
     - shape 最多只能有一个元素为 -1
       - shape 有 2 个-1：shape 为 `[-1,-1,2]`(异常)
       - shape 有 1 个-1：shape 为 `[-1,2,2]`
       - shape 没有-1：shape 为 `[2,2,2]`
     - axis 的数值应在`[-len(x.shape), len(x.shape)-1]`范围内
       - axis 为正数，在范围外：`x=paddle.randn([2,4,5])`, `axis=3`(异常)
       - axis 为负数，在范围外：`x=paddle.randn([2,4,5])`, `axis=-4`(异常)
       - axis 为负数，在范围内：`x=paddle.randn([2,4,5])`, `axis=-2`
       - axis 为正数，在范围内：`x=paddle.randn([2,4,5])`, `axis=1`
4. 边界测试：
   - x 是一个空的 Tensor，即 x.shape = (0,)
   - shape 是一个空的 tuple 或 list，即 shape = ()
   - shape 包含了 0 作为元素，例如 shape = (2, 0, 3)
   - shape 包含了负数作为元素（除了-1），例如 shape = (-2, 4)

# 七、可行性分析和排期规划

方案主要依赖现有 paddle api 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无

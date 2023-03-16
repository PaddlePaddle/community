# paddle.unflatten 设计文档

| API 名称     | Unflatten                            |
| ------------ | ------------------------------------ |
| 提交作者     | [cos43](https://github.com/cos43)    |
| 提交时间     | 2023-03-14                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | develop                              |
| 文件名       | 20230314_api_design_for_unflatten.md |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.unflatten，Tensor.unflatten，paddle.nn.Unflatten。

## 2、功能目标

实现 `unflatten` API，将输入Tensor的某一个维度，扩展成多个维度。

## 3、意义

为 paddle 框架中提供一种将输入Tensor的某一个维度，扩展成多个维度的API。

# 二、飞桨现状

飞桨中提供了多种形状变换的 API，包括但不限于：

`paddle.reshape`: 改变 Tensor 的形状，可以将 Tensor 展平成一维或将一维 Tensor 展开成多维。

`paddle.flatten`: 将 Tensor 展平成一维。

目前，飞桨暂时没有直接将 Tensor 某一个维度展开成多个维度的 API。如果需要展开某一维度，可以先s计算展开后的shape，然后使用 `paddle.reshape` 展开。

# 三、业内方案调研

## Pytorch

Pytorch 中相关的API如下：

`torch.unflatten(input, dim, sizes)`

支持在多个维度上展开输入张量的维度。

Parameters:

- **input** ([*Tensor*](https://pytorch.org/docs/2.0/tensors.html#torch.Tensor)) – the input tensor.
- **dim** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Dimension to be unflattened, specified as an index into `input.shape`.
- **sizes** (*Tuple* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]*) – New shape of the unflattened dimension. One of its elements can be -1 in which case the corresponding output dimension is inferred. Otherwise, the product of `sizes` *must* equal `input.shape[dim]`.

Returns:

A View of input with the specified dimension unflattened.

官方文档链接为：[torch.unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.unflatten.html?highlight=unflatten#torch.unflatten)



`torch.nn.Unflatten(dim, unflattened_size)`

展开一个张量，将其扩展到所需的形状。用于Sequential。

Parameters:

- **dim** (*Union* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str)*]*) – Dimension to be unflattened
- **unflattened_size** (*Union**[**torch.Size,* *Tuple*, *List*]) – New shape of the unflattened dimension

Shape:

- Input: (∗, $S_{dim}$ ,∗) where $S_{dim}$ is the size at dimension `dim` and ∗ means any number of dimensions including none.
- Output: (∗, $U_{1}$ ,..., $U_{n}$ ,∗), where $U$ = `unflattened_size` and $\prod_{i=1}^{n} U_i=S_{dim}$

官方文档链接：[Unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.nn.Unflatten.html?highlight=unflatten#torch.nn.Unflatten)



`Tensor.unflatten(dim, sizes)`

同`torch.unflatten(input, dim, sizes)`

官方文档链接：[torch.Tensor.unflatten — PyTorch 2.0 documentation](https://pytorch.org/docs/2.0/generated/torch.Tensor.unflatten.html?highlight=unflatten#torch.Tensor.unflatten)

#### Tensorflow

Tensorflow没有直接api，但是可以使用reshape达到相同的效果。

#### Numpy

Numpy没有直接api，但是可以使用reshape达到相同的效果。

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
- 使用torch.view_symint方法将输入张量调整为展开后的形状
- 返回输出张量

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.unflatten(x, sizes, axis)` 

参数说明如下：

- **x** (Tensor) – 要进行扩展的张量。
- **sizes** (*Tuple* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* | *List* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* ) – 扩展后张量的新形状。其中一个元素可以是-1，在这种情况下，将推断相应的输出维度。否则，`sizes`的乘积必须等于`input.shape[dim]`。
- **axis** (int) – 需要扩展张量的维度。

返回的是一个在axis维度扩展成sizes形状的tensor

`Tensor.unflatten(sizes, axis)` 

参数说明如下：

- **sizes** (*Tuple* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* | *List* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* ) – 扩展后张量的新形状。其中一个元素可以是-1，在这种情况下，将推断相应的输出维度。否则，`sizes`的乘积必须等于`input.shape[dim]`。
- **axis** (int) – 需要扩展张量的维度。

返回的是一个在axis维度扩展成sizes形状的tensor

`paddle.nn.Unflatten(sizes, axis)`

参数说明如下：

- **sizes** (*Tuple* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* | *List* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* ) – 扩展后张量的新形状。其中一个元素可以是-1，在这种情况下，将推断相应的输出维度。否则，`sizes`的乘积必须等于`input.shape[dim]`。
- **axis** (int) – 需要扩展张量的维度。

## 底层 OP 设计

使用 paddle现有 API 进行设计，不涉及底层OP新增与开发。

## API 实现方案
这个方法的目的是将一个张量（tensor）在指定的轴（axis）上展开为指定的形状（sizes）

- 检查输入参数是否合法，如果不合法，抛出相应的异常

- 检测sizes中是否有-1，如果有-1，表示需要根据输入张量的形状和其他维度推断出-1对应的值

- 判断sizes的积是否等于tensor.shape[axis]

- 计算扩展之后的shape，计算方法为:

  `shape = input_shape[:axis] + list(sizes) + input_shape[axis + 1:]`

  其中`input_shape = tensor.shape`

- 最后，这个方法使用paddle.reshape函数将输入张量转换为新的形状，并返回结果

# 六、测试和验收的考量

1. 结果正确性:

   - 这个函数的测试和验收的目标是确保它能正确地将输入张量在指定轴上展开为指定形状，并且能处理各种异常情况。
   - 前向计算: `paddle.unflatten`计算结果与 `torch.unflatten` 计算结果一致。
- 反向计算:由 Python 组合新增 API 无需验证反向计算。
2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。
3. 单元测试:

   - 数据类型检验:
     - input 要求为 paddle.Tensor
     - sizes 要求为 *Tuple* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* 或者 *List* *[*[*int*](https://docs.python.org/3/library/functions.html#int)*]* 
     - axis 要求为 int
   - 具体数值检验:
     - 对于 sizes, 则要求里面最多只有一个 -1
     - 对于 axis , 则要求 input存在该维度
   - 正常情况：给定合法的输入参数，检查输出张量是否符合预期的形状
     - 例如，给定input为一个形状为(2, 4, 4)的张量，axis为1，sizes为(2, -1)，则输出张量应该是一个形状为(2, 2, 2, 4)的张量
   - 异常情况：给定不合法的输入参数，检查是否抛出相应的异常，并检查异常信息是否正确
     - 例如，给定input为一个字符串"hello"，axis为1，sizes为(2, -1)，则应该抛出TypeError异常，并提示`Invalid input type: <class ‘str’>. Expected paddle.Tensor`
   - 边界情况：给定一些特殊或极端的输入参数，检查输出张量是否符合预期的形状
     - 例如，给定input为一个形状为(2, 4, 4)的张量，axis为0，sizes为(-1,)，则输出张量应该是一个形状为(2, 8)的张量

# 七、可行性分析和排期规划

方案主要依赖现有 paddle api 组合而成，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无

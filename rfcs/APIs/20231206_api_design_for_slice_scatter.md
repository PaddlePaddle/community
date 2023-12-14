# paddle_slice_scatter 设计文档

| API名称                                                      | paddle.slice_scatter                  |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | megemini (柳顺)                                    |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-12-13                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   |
| 文件名                                                       | 20231213_api_design_for_slice_scatter.md<br> |



# 一、概述
## 1、相关背景

为了提升飞桨API丰富度，需要为飞桨扩充API `paddle.slice_scatter`

本API属于飞桨开源个人贡献赛API开发任务[No.28：为 Paddle 新增 slice_scatter API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_5th/%E3%80%90PaddlePaddle%20Hackathon%205th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%20API%20%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no28%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-slice_scatter-api)的任务。

## 2、功能目标

对于一个Tensor，根据给定的轴和切片表示的索引范围，返回一个新的Tensor，其结果等价于将value 中的值填充到该Tensor上，例如当指定轴为1，索引为start=1, end=5, step=2的切片时，与x[:, 1:5:2] = value 结果相似，但与前者不同的是，不会直接修改x的值，而是返回预期赋值后的结果，即额外进行了拷贝操作

预期该API支持

- paddle.slice_scatter 作为独立的函数调用
- Tensor.slice_scatter，作为 Tensor 的方法使用

## 3、意义

为飞桨增加根据给定的轴和切片索引范围，返回替换掉部分切片索引值的新Tensor，提升飞桨API丰富度。

# 二、飞桨现状

目前飞桨缺少相关功能实现

# 三、业内方案调研

## PyTorch

PyTorch 中有 API `torch.slice_scatter(input, src, dim=0, start=None, end=None, step=1) → Tensor` 以及对应的`torch.Tensor.slice_scatter`

其介绍为：

> Embeds the values of the src tensor into input at the given dimension. This function returns a tensor with fresh storage; it does not create a view.
参数表为：
- `Tensor` input: the input tensor.
- `Tensor` src: The tensor to embed into input
- `int` dim: the dimension to insert the slice into
- `Optional[int]` start: the start index of where to insert the slice
- `Optional[int]` end: the end index of where to insert the slice
- `int` step: the how many elements to skip in

### 前向实现

PyTorch在通常的训练过程中通过调用ATen算子来实现相关操作，可以参考[aten/src/ATen/native/TensorShape.cpp#L3956](https://github.com/pytorch/pytorch/blob/34ded743998e2f6fa9677db0cb02658bcc657f05/aten/src/ATen/native/TensorShape.cpp#L3956)。

```cpp
at::Tensor slice_scatter(const at::Tensor& self, const at::Tensor& src, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
    // See Note [*_scatter ops preserve strides]
    auto output = clone_preserve_strides(self);
    auto slice = output.slice(dim, start, end, step);
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}
```
在Torch 2.0版本新推出的TorchDynamo中，其默认后端Inductor针对`slice_scatter`进行生成的代码如下，具体代码可以参考[torch/_inductor/lowering.py#L2108](https://github.com/pytorch/pytorch/blob/34ded743998e2f6fa9677db0cb02658bcc657f05/torch/_inductor/lowering.py#L2108)
```python
@register_lowering(aten.slice_scatter, type_promotion_kind=None)
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    assert x.get_dtype() == src.get_dtype()
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    if start is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(start, 0)):
        start = start + dim_size
    if end is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(end, 0)):
        end = end + dim_size
    if start is None:
        start = 0
    if end is None or V.graph.sizevars.statically_known_leq(x.get_size()[dim], end):
        end = dim_size
    src_size = list(x.get_size())
    src_size[dim] = FloorDiv(sympy.expand(end - start), sympy.expand(step))
    src = expand(src, src_size)
    src_loader = src.make_loader()
    def inner_fn(idx):
        if start == 0 and end == dim_size and step == 1:
            # selecting every element is the same as just src.clone()
            return src_loader(idx)
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        src_idx = list(idx)
        src_idx[dim] = FloorDiv(idx[dim] - start, step)
        mask = []
        if start != 0:
            mask.append(
                ops.ge(
                    idx_dim,
                    ops.index_expr(sympy.expand(start), torch.int64),
                )
            )
        if end != dim_size:
            mask.append(
                ops.lt(
                    idx_dim,
                    ops.index_expr(sympy.expand(end), torch.int64),
                )
            )
        if step != 1:
            mask.append(
                ops.eq(
                    ops.index_expr(
                        ModularIndexing(idx[dim] - start, 1, step), torch.int64
                    ),
                    ops.constant(0, torch.torch.int64),
                )
            )
        assert mask
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(
            mask,
            lambda: src_loader(src_idx),
            0 if is_integer_type(x) else 0.0,
        )
        return ops.where(
            mask,
            src_val,
            x_loader(idx),
        )
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )
```

### 反向实现

PyTorch的Autograd中定义了相应的反向实现，可以参考[tools/autograd/derivatives.yaml#L1491](https://github.com/pytorch/pytorch/blob/34ded743998e2f6fa9677db0cb02658bcc657f05/tools/autograd/derivatives.yaml#L1491)

```yaml
- name: slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
  self: slice_scatter_symint(grad, zeros_like(src), dim, start, end, step)
  src: grad.slice_symint(dim, start, end, step)
  result: auto_linear
```

可以看到，对于两个输入的反向梯度均有相关定义描述

## MindSpore

MindSpore 中有 `mindspore.ops.slice_scatter` 此接口：

- `mindspore.ops.slice_scatter(input, src, axis=0, start=None, end=None, step=1)`

其实现代码：

https://www.mindspore.cn/docs/zh-CN/r2.1/_modules/mindspore/ops/function/array_func.html#slice_scatter

``` python
def _get_slice_scatter_const(x_shape, axis, start, end, step):
    r"""
    Calculate the rank of input, embedded dimensions and index.
    """
    x_rank = len(x_shape)
    axis = axis if axis >= 0 else axis + x_rank
    start = start if start is not None else 0
    start = start if start >= 0 else start + x_rank
    end = end if end is not None else x_shape[axis]
    end = end if end >= 0 else end + x_rank
    end = end if end < x_shape[axis] else x_shape[axis]
    index = list(builtins.range(start, end, step))
    return x_rank, index, axis


[文档]def slice_scatter(input, src, axis=0, start=None, end=None, step=1):
    r"""
    Slice the input Tensor in the specified dimension and overlay the slice results with the source Tensor.
    The `input` is sliced along the specified dimension. The start position of the slice is `start` ,
    the end position is `end` , and the step size is `step` .
    Then the slicing result is overwritten with `src` to get the output Tensor.

    Args:
        input (Tensor): The target Tensor.
        src (Tensor): The source Tensor.
        axis (int, optional): The dimension of `input` to be sliced. Default: ``0`` .
        start (int, optional): The start index to slice in the specified dimension.
            Default: ``None``, `start` is ``0`` .
        end (int, optional): The end index to slice in the specified dimension.
            Default: ``None``, `end` is the length of `input` in the specified dimension.
        step (int, optional): Step size. Default: ``1``, the distance from the next slice element is ``1`` .

    Returns:
        Tensor after embedding, has the same shape and type as `input` .

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `src` is not a Tensor.
        TypeError: If `axis` or `step` is not an integer.
        TypeError: If `start` or `end` is not ``None`` or an integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> a = ms.ops.zeros((4, 6))
        >>> b = ms.ops.ones((4, 3))
        >>> output = ms.ops.slice_scatter(a, b, axis=1, start=0, end=5, step=2)
        >>> print(output)
        [[1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]]
    """
    input_shape = input.shape
    input_rank, index, axis = _get_slice_scatter_const(input_shape, axis, start, end, step)

    src_shape = src.shape
    index_shape = input_shape[:axis] + (len(index),) + input_shape[axis + 1:]
    index_tensor = ms.Tensor(index)
    for _ in builtins.range(axis):
        index_tensor = index_tensor.expand_dims(0)

    if index_shape == src_shape:
        for _ in builtins.range(input_rank - axis - 1):
            index_tensor = index_tensor.expand_dims(-1)
        index_tensor = index_tensor.broadcast_to(src.shape)
        return tensor_scatter_elements(input, axis=axis, indices=index_tensor, updates=src)

    for _ in builtins.range(axis):
        src = src.expand_dims(0)
    if axis == input_rank - 1:
        src = src.broadcast_to(input.shape[0:axis] + src_shape)
    else:
        for _ in builtins.range(len(src_shape)):
            index_tensor = index_tensor.expand_dims(-1)
        src = src.broadcast_to(input.shape[0:axis] + (len(index),) + src_shape)
    index_tensor = index_tensor.broadcast_to(src.shape)
    output = tensor_scatter_elements(input, axis=axis, indices=index_tensor, updates=src)
    return output

```


# 四、对比分析

对比 PyTorch 与 MindSpore:

- 实现方式不同

  PyTorch 通过 c++ 实现；MindSpore 通过 python 实现。

- 实现逻辑不同

  PyTorch 使用的是 slice，将输入分割后 copy src 的值； MindSpore 使用的是 scatter，将输入使用 scatter 根据 src 进行更新。


# 五、设计思路与实现方案

paddle 目前的 `set_value` 算子已经支持 `axes`, `starts`, `ends`, `steps` 等参数，因此，可以使用 `set_value` 算子实现 `slice_scatter` ，由于要求输入 `x` 与 `values` 具有相同的 `ndim`，因此，不需要使用 `decrease_axes` 等参数。

## 命名与参数设计

添加 Python API:
```python
paddle.slice_scatter(x, values, axis=0, start=None, stop=None, step=1, name=None)
```

参数表：

- x: (Tensor) 输入的 tensor。数据类型支持 `float32`、`float64`。
- values: (Tensor) 用于填充的 tensor。数据类型与input一致，形状与`x[*x.shape[:axis], start:end:step, *x.shape[axis+1:]]`取出的slice一致。
- axis: (int) y的数据将被填充至x的axis维度。
- start: (Optional[int]) 待插入slice位置的起始index。
- stop: (Optional[int]) 待插入slice位置的结束index。
- step: (int) 待插入slice的步长。
- name: (Optional[str]) op 名称

## 底层OP设计

不涉及底层 OP。

## API实现方案

此次使用 `set_value` 算子实现接口：

``` python
def slice_scatter(x, values, axis=0, start=None, stop=None, step=1, name=None):
    
    if x.ndim != values.ndim:
        raise ValueError(
            f"The input x and values should have save dimension, but got input of {x.ndim} and values of {values.ndim}."
        )

    x_shape = x.shape
    values_shape = values.shape

    index = list(range(start or 0, stop or x_shape[axis], step))
    exp_shape = [*x_shape[:axis], len(index), *x_shape[axis+1:]]
    if exp_shape != values_shape:
        raise ValueError(
            "The values.shape should be same of [*x_shape[:axis], len(index), *x_shape[axis+1:]],"
            f"but got values.shape of {values.shape} and slice shape {exp_shape}."
        )

    starts = [start]
    ends = [stop]
    steps = [step]
    axes = [axis]
    none_axes = []
    decrease_axes = []
    inputs = {'Input': x}
    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes,
        'none_axes': none_axes,
    }

    dtype = x.dtype
    attrs['dtype'] = dtype

    values = values.astype(dtype)
    inputs["ValueTensor"] = values

    if in_dynamic_or_pir_mode():
        return _C_ops.set_value_with_tensor(
            x,
            values,
            starts,
            ends,
            steps,
            axes,
            decrease_axes,
            none_axes,
        )
    else:
        helper = LayerHelper('slice_scatter', **locals())
        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        cur_block = default_main_program().current_block()
        cur_block.append_op(
            type="set_value",
            inputs=inputs,
            outputs={'Out': output},
            attrs=attrs,
            inplace_map={"Input": "Out"},
        )

        return output
```

有几点说明：

- x 与 src 需要有相同的 ndim
- values_shape 需要与 slice 的 exp_shape 一致
- 参数 axis/start/stop/step 不支持 list。因为，多个 axis 的话可能导致 slice 的 shape 错误。
  比如，x 为 [8, 8], src 为 [8, 2]，则 axis 只能为 1。


# 六、测试和验收的考量

- 覆盖动态图和静态图的测试场景
- 覆盖 CPU、GPU 两种测试场景
- 支持各种Tensor精度，FP32、FP64（带验证）
- 需要检查前向和反向计算的精度正确性
- 处理0维输入数据
- 处理可选参数不存在或不一致的情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关PythonAPI均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

# 名词解释

# 附件及参考资料

[【Hackathon 5th No.28】为 Paddle 新增 slice_scatter API](https://github.com/PaddlePaddle/community/pull/668)
[PyTorch slice_scatter 文档](https://pytorch.org/docs/stable/generated/torch.slice_scatter.html)
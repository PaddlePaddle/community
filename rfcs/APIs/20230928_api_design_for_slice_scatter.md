# paddle.slice_scatter 设计文档

|API名称 | paddle.slice_scatter | 
|---|---|
|提交作者 | VOIDMalkuth | 
|提交时间 | 2023-09-28 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20200928_api_design_for_slice_scatter.md | 


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

# 四、对比分析

Torch中默认使用C++实现算子，并且手动添加了对应的反向信息等。但C++中的算子仍然以Tensor级别进行操作，调用的也是Tensor的API完成操作，因此完全可以在Python中实现，并且使用autograd技术自动完成反向操作。而Torch的Inductor直接生成IR则仅适用于TorchDynamo场景。

# 五、设计思路与实现方案

## 命名与参数设计

添加 Python API:
```python
paddle.slice_scatter(x, y, axis=0, start=None, stop=None, step=1)
```

参数表：

- x: (Tensor) 输入的 tensor。数据类型支持 `float16`、`float32`、`float64`、`int8`、`int16`、`int32`、`int64`、`bfloat16`。
- y: (Tensor) 用于填充的 tensor。数据类型与input一致，形状与`x[*x.shape[:axis], start:end:step, *x.shape[axis+1:]]`取出的slice一致。
- axis: (int) y的数据将被填充至x的axis维度。
- start: (Optional[int]) 待插入slice位置的起始index。
- stop: (Optional[int]) 待插入slice位置的结束index。
- step: (int) 待插入slice的步长。

## 底层OP设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案

- 使用`paddle.Tensor.clone`拷贝一个Tensor
- 使用PythonAPI组合，以索引或切片方式在新Tensor上赋值以完成功能。

# 六、测试和验收的考量

- 覆盖动态图和静态图的测试场景
- 覆盖 CPU、GPU 两种测试场景
- 支持各种Tensor精度，FP32、FP64、FP16、INT8、INT16、INT32、INT64
- 需要检查前向和反向计算的精度正确性，但由于Python 组合方式新增的 API 反向计算已经在各组合 API 单测中分别验证了，因此不必过多关注
- 处理0维输入数据
- 处理可选参数不存在或不一致的情况

# 七、可行性分析和排期规划

有业内方案实现作为参考，相关PythonAPI均有实现，可以在开源贡献个人挑战赛期间完成。

# 八、影响面

对其他模块暂无影响

# 名词解释

# 附件及参考资料

[PyTorch slice_scatter 文档](https://pytorch.org/docs/stable/generated/torch.slice_scatter.html)

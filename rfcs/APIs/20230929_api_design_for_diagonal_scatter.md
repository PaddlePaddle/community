# paddle.diagonal_scatter 设计文档

| API名称      | paddle.diagonal_scatter                     |
| ------------ | ------------------------------------------- |
| 提交作者     | DanGuge                                     |
| 提交时间     | 2023-09-29                                  |
| 版本号       | V1.0                                        |
| 依赖飞桨版本 | develop                                     |
| 文件名       | 20230929_api_design_for_diagonal_scatter.md |


 # 一、概述
 ## 1、相关背景
丰富Paddle的Tensor相关API，支持更多样的tensor操作

 ## 2、功能目标
实现diagonal_scatter API，能够将src张量值嵌入到input张量中，同时src张量将会沿着input的对角线元素分布，支持dim1和dim2两个维度：

- paddle.diagonal_scatter作为独立函数调用
- Tensor.diagonal_scatter作为Tensor的方法使用

 ## 3、意义
支持Paddle在张量上执行更细粒度的操作

 # 二、飞桨现状
目前飞桨缺少相关功能的实现，但是可以通过飞桨的API paddle.diagonal组合其他函数修改对应位置的值

 # 三、业内方案调研

 ## PyTorch
PyTorch中有API：`torch.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1)`

在PyTorch中介绍如下，功能与Paddle的需求一致：

```
Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.
This function returns a tensor with fresh storage; it does not return a view.
The argument offset controls which diagonal to consider:

* If offset = 0, it is the main diagonal.
* If offset > 0, it is above the main diagonal.
* If offset < 0, it is below the main diagonal.
```

### 实现方法

在实现方法上，Pytorch的API torch.diagonal_scatter基于C++ API组合实现了此功能，核心代码如下：

- diagonal_scatter的核心功能依赖于diagonal
    - 首先，对intput进行深度拷贝，获得output
    - 接着，调用diagonal函数获得output对应的diagonal切片
    - 最后，将src填充到diagnoal切片中，返回output

```cpp
// pytorch/aten/src/ATen/native/TensorShape.cpp
at::Tensor diagonal_scatter(const at::Tensor& self, const at::Tensor& src, int64_t offset, int64_t dim1, int64_t dim2) {
    // See Note [*_scatter ops preserve strides]
    auto output = clone_preserve_strides(self);
    auto slice = output.diagonal(offset, dim1, dim2);
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}
```

```cpp
// pytorch/aten/src/ATen/native/TensorShape.cpp
Tensor diagonal(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim();
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  NoNamesGuard no_names_guard;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t diag_size;
  int64_t storage_offset = self.storage_offset();
  // compute storage offset and size for the diagonal
  // for positive values of offset (above the main diagonal)
  // "leftmost columns" (along dim2) are dropped
  // for negative values of offset (below the main diagonal)
  // "topmost rows" (along dim1) are dropped.
  // Note that we invert +/- in the second to absorb the negative
  // sign in the offset.
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(self.size(dim1), self.size(dim2)-offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(self.size(dim1)+offset, self.size(dim2)), 0);
  }

  // NumPy allows you to specify offsets "off the end"; let's just be careful not to
  // set a ridiculous storage_offset in that case (technically it shouldn't matter
  // because there are no elements in the tensor, but let's be kosher).
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * self.stride(dim2);
  } else {
    storage_offset -= offset * self.stride(dim1);
  }

  // construct new size and stride: we drop dim1 and dim2 (maximum first for not changing the index of the minimum)
  // the new ("joint") dimension is appended to the end of the shape / stride to match numpy semantics
  DimVector sizes(self.sizes().begin(), self.sizes().end());
  DimVector strides(self.strides().begin(), self.strides().end());
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1)+self.stride(dim2));

  // return view with new parameters
  auto result = self.as_strided(sizes, strides, storage_offset);

  no_names_guard.reset();
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}
```

```cpp
// Clones a tensor by cloning the underlying storage that it came from,
// which allows us to replicate the exact strides/storage_offset in the cloned tensor.
// Note [*_scatter ops preserve strides]
// In order for functionalization to preserve stride correctness, the *_scatter
// operators that it calls must preserve the striding behavior of their inputs.
// Specifically, the output of *_scatter(base, mutated_view, ...)
// should have identical size/stride/storage_offset to "base".
at::Tensor clone_preserve_strides(const at::Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.has_storage());
  // In cases where the input tensor has internal memory overlap, we cannot actually
  // preserve the strides/storage_offset of the input tensor, because
  // *_scatter ops will try to copy_() into the cloned tensor.
  // However, this should **never** show up in functionalized user code;
  // most aten ops that try to mutate a tensor with internal memory overlap would error anyway.
  //
  // The one place that this does come up is in autograd - if there's a select_scatter
  // in the forward, then autograd will generate one for the backward.
  // If the input to the select_scatter is grad_output, then this could be an expanded tensor
  // with internal overlap.
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    return self.clone();
  }
  auto dtype_size = self.dtype().itemsize();
  auto nbytes = self.storage().sym_nbytes();
  TORCH_INTERNAL_ASSERT(nbytes % dtype_size == 0);
  auto numel = nbytes / dtype_size;
  auto self_full_size = self.as_strided_symint({std::move(numel)}, {1}, 0);
  auto clone = self_full_size.clone();
  auto out = clone.as_strided_symint(self.sym_sizes(), self.sym_strides(), self.sym_storage_offset());
  return out;
}
```

* 在PyTorch2.0的新编译器中，默认后端inductor也对diagonal_scatter进行了实现

```python
@register_lowering(aten.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input, src, offset: int = 0, dim1: int = 0, dim2: int = 1):
    output = clone(input)
    target = diagonal(output, offset, dim1, dim2)
    mutate_to(target, src)
    return output

@register_lowering(aten.diagonal, type_promotion_kind=None)
def diagonal(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    original_shape = input.get_size()
    num_dims = len(original_shape)
    dim1 = canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = canonicalize_dim(idx=dim2, rank=num_dims)

    check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    offset_negative = V.graph.sizevars.evaluate_expr(sympy.Lt(offset, 0))
    if offset_negative:
        diag_size = max(min(original_shape[dim1] + offset, original_shape[dim2]), 0)
    else:
        diag_size = max(min(original_shape[dim1], original_shape[dim2] - offset), 0)

    base_idx = (0, 0)
    if offset_negative:
        base_idx = (-offset, 0)
    else:
        base_idx = (0, offset)

    sizes = [s for i, s in enumerate(original_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    def reindexer(idx):
        diag_idx = idx[-1]
        original_idx = [0] * len(original_shape)
        cur_dim = 0
        for d in range(num_dims):
            if d == dim1:
                original_idx[d] = diag_idx + base_idx[0]
            elif d == dim2:
                original_idx[d] = diag_idx + base_idx[1]
            else:
                original_idx[d] = idx[cur_dim]
                cur_dim += 1

        assert cur_dim == len(original_shape) - 2
        return original_idx

    return TensorBox(ir.GenericView.create(input, sizes, reindexer))
```

 ## TensorFlow

TensorFlow中没有diagonal_scatter API的实现，但是有核心函数tf.linalg.diag，可以通过组合来实现对应逻辑
 ## Numpy

Numpy中没有diagonal_scatter API的实现，但是有核心函数numpy.diagonal，可以通过组合来实现对应逻辑

 # 四、对比分析
  - PyTorch是在C++ API基础上实现，使用Python调用C++对应的接口
  - Tensorflow、Numpy中没有对应API的实现


 # 五、设计思路与实现方案

 ## 命名与参数设计
 ```python
paddle.diagonal_scatter(input, src, offset=0, axis1=0, axis2=1, name=None)
Tensor.diagonal_scatter(input, src, offset=0, axis1=0, axis2=1, name=None)
 ```
参数定义：

- `input(Tensor)`：输入张量，张量的维度至少为2维
- `src(Tensor)`：嵌入张量，将会被嵌入到输入张量中
- `offset(int, optional)`：偏移的对角线
    - 偏移量为0，则嵌入对角线位置
    - 偏移量大于0，则嵌入对角线上方
    - 偏移量小于0，则嵌入对角线下方
- `axis1(int, optional)`：对角线的第一个维度，默认值为0
- `axis2(int, optional)`：对角线的第二个维度，默认值为1
- `name (str，optional)`：具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None

 ## 底层OP设计

依赖已有的API（diagonal）实现，无需实现新的底层OP

 ## API实现方案
在python/paddle/tensor/manipulation.py中增加diagonal_scatter函数

通过`paddle.diagonal`实现对应逻辑

```python
output = deep_clone(input)
slice = paddle.diagonal(output, offset, axis1, axis2)
slice.copy_(src)
```

## 代码实现文件路径

函数API实现路径：python/paddle/tensor/manipulation.py

单元测试路径：test/lagacy_test/test_diagonal_scatter.py

 # 六、测试和验收的考量

测试考虑以下case：

- 校验diagonal_scatter答案的正确性，对比torch.diagonal_scatter进行校验

- 检查参数的正确性，比如是否为支持的数据类型，是否在offset/axis1/axis2设置有误时进行报错

- 检查input的维度是否符合大于等于2个维度
- 检查input的slice和src的维度是否相等，这样才能进行覆盖

 # 七、可行性分析和排期规划

 方案实施难度可控，工期上可以满足在当前版本周期内开发完成


 # 八、影响面

 为已有 API 的增强，对其他模块无影响。


 # 名词解释

 无

 # 附件及参考资料

[torch.diagonal_scatter](https://pytorch.org/docs/stable/generated/torch.diagonal_scatter.html)


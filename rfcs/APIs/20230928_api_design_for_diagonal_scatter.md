# 标题 paddle.diagonal_scatter 设计文档

|API名称 | paddle.diagonal_scatter | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 吴俊([bapijun] (https://github.com/bapijun)) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-29 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发 | 
|文件名 | 20230923_api_design_for_diagonal_scatter.md | 


# 一、概述
## 1、相关背景
丰富Paddle的Tensor相关API，支持更多样的tensor操作

## 2、功能目标

对于一个Tensor，对于tensor a 和 b，将 b 中的内容按照索引的位置嵌入 a 中。如索引偏移量为0，则嵌入对角线位置。如索引偏移量 >0，则嵌入对角线上方，如偏移量 <0，则嵌入对角线下方。例如a = paddle.zeros([2,2])，b= paddle.ones([2])，输出为\[[1.0,0.0],[0.0,1.0]]
调用路径：
paddle.diagonal_scatter 作为独立的函数调用
Tensor.diagonal_scatter，作为 Tensor 的方法使用

## 3、意义

为 Paddle 新增 `paddle.diagonal_scatter` API，丰富Paddle的Tensor相关API，支持更多样的tensor操作

# 二、飞桨现状

目前飞桨框架并不存在对应的api，可以通过其他的代码实现


# 三、业内方案调研

### 1. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1)`

- `input` 为 输入tensor,至少是2维。
- `src` 为 tensor类型，用于填充input。
- `offset` int类型,可选,决定是哪一个对角线,默认为0。
- `dim1` int类型,可选,第一个维度来考虑对角线,默认为0。
- `dim1` int类型,可选,第而个维度来考虑对角线,默认为1。
其实现的代码如下

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

### 2. TensorFlow

没有找到对应的api

### 3. MindSpore

在 MindSpore 中使用的 API 格式如下：

`mindspore.ops.diagonal_scatter`

函数定义与pytorch类似

其实现的代码如下
``` python
def _check_diagonal_scatter_shape(diag_shape, src_shape):
    if diag_shape != src_shape:
        raise ValueError(f"For diagonal_scatter, the shape of src should equal to the shape of input diagonal,"
                         f"but got src.shape {src_shape} and diagonal shape {diag_shape}.")


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
   
    _check_is_tensor("input", input, "diagonal_scatter")
    _check_is_tensor("src", src, "diagonal_scatter")
    _check_is_int(offset, "offset", "diagonal_scatter")
    _check_is_int(dim1, "dim1", "diagonal_scatter")
    _check_is_int(dim2, "dim2", "diagonal_scatter")
    input_diag = input.diagonal(offset, dim1, dim2)
    _check_diagonal_scatter_shape(input_diag.shape, src.shape)
    embed = ones_like(src)
    embed = ops.diag_embed(embed, offset, dim1, dim2)
    embed = input * embed
    src = ops.diag_embed(src, offset, dim1, dim2)
    return input + src - embed
```


# 四、对比分析

这里倾向于pytorch和MindSpore的方式，其二者的实现思路类型,都是采用diagonal切片后,针对代码进行填入后才做处理,这里采用和MindSpore类似的方式去实现。

# 五、设计思路与实现方案

## 命名与参数设计

paddle.diagonal_scatter

 ```python
paddle.diagonal_scatter(x, y, offset=0, dim1=0, dim2=1, name=None)
 ```
参数定义：

- `x(Tensor)`：输入张量，张量的维度至少为2维
- `y(Tensor)`：嵌入张量，将会被嵌入到输入张量中
- `offset(int, optional)`：偏移的对角线
    - 偏移量为0，则嵌入对角线位置
    - 偏移量大于0，则嵌入对角线上方
    - 偏移量小于0，则嵌入对角线下方
- `dim1(int, optional)`：对角线的第一个维度，默认值为0
- `dim2(int, optional)`：对角线的第二个维度，默认值为1
- `name (str，optional)`：具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None


Tensor.diagonal_scatter

```python
Tensor.diagonal_scatter(x, offset=0, dim1=0, dim2=1, name=None)

```
参数定义：

- `x(Tensor)`：嵌入张量，将会被嵌入到输入张量中
- `offset(int, optional)`：偏移的对角线
    - 偏移量为0，则嵌入对角线位置
    - 偏移量大于0，则嵌入对角线上方
    - 偏移量小于0，则嵌入对角线下方
- `dim1(int, optional)`：对角线的第一个维度，默认值为0
- `dim2(int, optional)`：对角线的第二个维度，默认值为1
- `name (str，optional)`：具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None

## 底层OP设计

依赖已有的API（fill_diagonal_tensor或diagonal）实现，无需实现新的底层OP

## API实现方案

参考MindSpore的方式去是实现对应的代码

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

可考虑一下场景：

测试考虑以下case：

- 校验diagonal_scatter答案的正确性，对比torch.diagonal_scatter进行校验

- 检查参数的正确性，比如是否为支持的数据类型，是否在offset/dim1/dim2设置有误时进行报错

- 检查input的维度是否符合大于等于2个维度

- 检查input的shape和src的维度是否是否能完成覆盖

# 七、可行性分析和排期规划
方案实施难度可控，工期上可以满足在当前版本周期内开发完成

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料

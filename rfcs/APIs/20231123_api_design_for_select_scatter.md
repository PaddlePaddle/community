# select_scatter API 设计文档

|API名称 | select_scatter |
|---|---|
|提交作者 | YibinLiu666 |
|提交时间 | 2023-11-23 |
|版本号 | 1.0                                       |
|依赖飞桨版本 | develop版本开发 |
|文件名 | 20231123_api_design_for_select_scatter.md |


# 一、概述
## 1、相关背景
`select_scatter` 是一个常用的API， 而paddle目前还没有该接口，因此需要为paddle新增`select_scatter`。

## 2、功能目标

实现`select_scatter` api， 根据给定的轴和特定索引位置，返回一个新的Tensor，其结果等价于将value 中的值填充到该Tensor上。例如当指定轴为1，索引位置为2时，与x[:, 2] = value 结果相似，但不会直接修改x的值，而是返回预期赋值后的结果。

- paddle.select_scatter 作为独立的函数调用
- Tensor.select_scatter，作为 Tensor 的方法使用

## 3、意义
提供`select_scatter`接口而不需要使用paddle的接口组合。

# 二、飞桨现状
目前可以通过api组合来实现该算子（参考PR  [PaddlePaddle/community: PaddlePaddle Developer Community (github.com)](https://github.com/PaddlePaddle/community/pull/664/files) ）

```python
import paddle
import numpy
def select_scatter(x, value, dim, index):
    indices = [slice(None)] * len(x.shape)
    indices[dim] = index
    x[tuple(indices)] = value
 x = paddle.to_tensor([[0, 0],
                      [0, 0]])
value = paddle.to_tensor([1, 2])
output = select_scatter(x, value, dim=0, index=0)
# [[1, 2],
#  [0, 0]]
```
此外，paddle底层已经实现了`set_value_with_tensor`以及`set_value`两个c op，能够通过start，stop，以及step等参数来设置Tensor指定位置的值。

# 三、业内方案调研
### pytorch

pytorch有相关的api kernel实现

```c++
at::Tensor select_scatter_symint(const at::Tensor& self, const at::Tensor& src, int64_t dim, c10::SymInt index) {
    auto output = clone_preserve_strides(self);
    auto slice = output.select_symint(dim, std::move(index));
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    slice.copy_(src);
    return output;
}

Tensor select_symint(const Tensor& self, int64_t dim, c10::SymInt index) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.sym_sizes()[dim];
  if (size < -index || size <= index) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      TORCH_CHECK_INDEX(false, "select(): index ", index, " out of range for tensor of size ",
                     self.sizes(), " at dimension ", self.names()[dim]);
    }
    TORCH_CHECK_INDEX(false, "select(): index ", index, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    return select_sparse(self, dim, index.guard_int(__FILE__, __LINE__));
  }

  Tensor result;
  if (self.is_quantized()) {
    auto local_index = index.guard_int(__FILE__, __LINE__);

    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    auto storage_offset = self.storage_offset() + local_index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    auto quantizer = create_subtensor_quantizer(self, true, local_index, local_index + 1, dim, 1);
    result = as_strided_qtensorimpl(self, sizes, strides, storage_offset, std::move(quantizer));
  } else {
    std::vector<c10::SymInt> sizes(self.sym_sizes().begin(), self.sym_sizes().end());
    std::vector<c10::SymInt> strides(self.sym_strides().begin(), self.sym_strides().end());
    auto storage_offset = self.sym_storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    result = self.as_strided_symint(sizes, strides, storage_offset);
  }
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}
```

## Tensorflow

Tensorflow 没有提供 `select_scatter` 的API。

# 四、对比分析

 PyTorch 是使用 C++ API 实现的，Python 端直接调用 C++ 接口，性能较好。尽管paddle能够通过算子组合实现该api，但是使用slice来 setitem 性能较差，并且无法达到非inplace的效果。因此计划在实现paddle的`select_scatter`时不直接使用slice来 setitem，而是根据输入来定义setitem使用的底层C op `set_value_with_tensor`所需要的输入和属性，直接调用该接口。

# 五、设计思路与实现方案

## 命名与参数设计
在python/paddle/tensor/manipulation.py添加python API

```python
def select_scatter(x, value, axis, index, name=None)->Tensor
```

其中

 **x**(Tensor) - 输入 Tensor , 支持 `bool`、`float16`、`float32`、`float64`、`uint8`、`int8`、`int16`、`int32`、`int64`、`bfloat16`、`complex64`、`complex64`。

 **value**(Tensor) - 需要嵌入到输入 Tensor 的 Tensor 值 , 支持 `bool`、`float16`、`float32`、`float64`、`uint8`、`int8`、`int16`、`int32`、`int64`、`bfloat16`、`complex64`、`complex64`。

**axis** (int) – 需要嵌入到src Tensor的维度。

**index** (int) – 选择的索引。

**name** (str|None): 层的名称(optional)。

## 底层OP设计

直接调用paddle现有的`set_value_with_tensor`的C op。



## API实现方案

根据输入的axis和index预处理好 `set_value_with_tensor` 所需要的 `start, stop, step, axis` 等输入参数，如何直接调用 `set_value_with_tensor` 将`src`的第`axis`维的`index`的数据改成`value`

# 六、测试和验收的考量
- 覆盖动态图和静态图的测试场景。

- 覆盖 CPU、GPU 两种测试场景。

- 需要保证前向计算、反向计算的精度正确性

   \+ 前向计算：通过 numpy 实现的函数的对比结果

   \+ 反向计算：通过 numpy 推导，计算反向结果的正确性

- 数据类型的测试：float64、float32、bfloat16等。

- 错误测试：src删除axis维度后与value的形状不一致。

# 七、可行性分析和排期规划
直接调用`set_value_with_tensor`算子的kernel，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料

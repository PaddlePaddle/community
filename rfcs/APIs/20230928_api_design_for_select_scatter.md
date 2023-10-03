# 新增 select_scatter API 设计文档

| API名称      | select_scatter                            |
| ------------ | ----------------------------------------- |
| 提交作者     | wyq-carol                                 |
| 提交时间     | 2023-09-28                                |
| 版本号       | V1.1                                      |
| 依赖飞桨版本 | develop                                   |
| 文件名       | 20230928_api_design_for_select_scatter.md |


 # 一、概述
 ## 1、相关背景
 为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API `paddle.select_scatter` 。

 ## 2、功能目标
 增加API `paddle.select_scatter` ，实现根据给定轴和特定索引位置，返回新Tensor。

 ## 3、意义
 可以支持在张量上执行非常细粒度的操作。

 # 二、飞桨现状
 目前飞桨的API缺少相关功能实现，只能通过paddle现有API组合实现。

```python
import paddle
import numpy
def select_scatter(x, value, dim, index):
    # 初始化每个维度上的切片，表示完整切片
    indices = [slice(None)] * len(x.shape)
    # 将dim的切片替换为index，用于在特定维度上定位元素
    indices[dim] = index
    # 使用tuple(indices)构造索引元组，将指定维度上的元素设置为value
    x[tuple(indices)] = value
    
x = paddle.to_tensor([[0, 0],
                      [0, 0]])
value = paddle.to_tensor([1, 2])
output = select_scatter(x, value, dim=0, index=0)
# [[1, 2],
#  [0, 0]]
```

使用 `slice` 可以完成 `select_scatter` API，且对各种数据类型通用 `bool`，`float16`，`float32`，`float64`，`int32`，`int64` 。

 # 三、业内方案调研

 ## PyTorch
 ### 实现方法
 Pytorch的API `torch.select_scatter` 基于C++ API组合实现了此功能，其中核心代码如下：

```c++
// pytorch/aten/src/ATen/native/TensorShape.cpp
Tensor select_scatter(const at::Tensor& self, const at::Tensor& src, int64_t dim, c10::SymInt index) {
	auto output = clone_preserve_strides(self);
    auto slice = output.select_symint(dim, std::move(index));
    TORCH_CHECK(slice.sizes() == src.sizes(), "expected src to have a size equal to the slice of self. src size = ", src.sizes(), ", slice size = ", slice.sizes());
    // 将维度匹配的src，填入self的slice位置
    slice.copy_(src);
    return output;
}
```

```c++
// pytorch/aten/src/ATen/native/TensorShape.cpp
Tensor select_symint(const Tensor& self, int64_t dim, c10::SymInt index) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  //维度匹配检查
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
  // 对稀疏张量有额外支持
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
    // slice 位置核心计算
    auto storage_offset = self.sym_storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    result = self.as_strided_symint(sizes, strides, storage_offset);
  }
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

```


 ## TensorFlow
TensorFlow 中没有 `select_scatter` API 的实现。
 ## Numpy

Numpy 中没有 `select_scatter` API 的实现。

 # 四、对比分析
  - PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口
  - Tensorflow、Numpy中没有 `select_scatter` API 的实现


 # 五、设计思路与实现方案

 ## 命名与参数设计
 ```python
paddle.select_scatter(x, src, dim, index)

Tensor.select_scatter(src, dim, index)

paddle.select_scatter_(x, src, dim, index)

Tensor.select_scatter_(src, dim, index)
 ```
* `x (Tensor)` 表示给定张量，支持数据类型 `bool`，`float16`，`float32`，`float64`，`int32`，`int64` 
* `src (Tensor)` 表示被填充等价于 `value` 的张量，支持与 `x` 相同的数据类型（可以实现 `int` 向 `float` 的转化以及 `float` 之间的转化）
* `dim (int)` 表示指定轴
* `index (int)` 表示索引位置
* `select_scatter_` 支持 `inplace` 方式修改输入张量

 ## 底层OP设计

依赖已有的 API 实现，不再单独设计 OP。

 ## API实现方案
函数实现路径：在 `python/paddle/tensor/manipulation.py` 中增加 `select_scatter` 函数

单元测试路径：在 `Paddle repo` 的 `test/legacy_test` 目录下增加 `test_select_scatter.py` 

初步实现方案如下：

```python
    # 初始化每个维度上的切片，表示完整切片
    indices = [slice(None)] * len(x.shape)
    # 将dim的切片替换为index，用于在特定维度上定位元素
    indices[dim] = index
    # 使用tuple(indices)构造索引元组，将指定维度上的元素设置为value
    x[tuple(indices)] = value
```

 # 六、测试和验收的考量

 基础测试：
- 计算输出数值结果的一致性和数据类型是否正确，使用 pytorch 作为参考标准
- 测试反向传播能否正常进行

边界测试：

- 张量 `x`，张量 `src` 的数据类型是否一致

* 张量 `x` 是否非零维
* 张量 `src` 与张量 `x` 的 slice 维度是否一致 

 # 七、可行性分析和排期规划

 方案实施难度可控，工期上可以满足在当前版本周期内开发完成。


 # 八、影响面

 为已有 API 的增强，对其他模块无影响。


 # 名词解释

 无。

 # 附件及参考资料

[torch.select_scatter — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.select_scatter.html?highlight=select_scatter#torch.select_scatter)


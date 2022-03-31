# paddle.sparse.sqrt 设计文档

| API名称                                                      | 新增API名称                                       |
| ------------------------------------------------------------ | ------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | paddle.sparse.sqrt                                |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-31                                        |
| 版本号                                                       | V1.0                                              |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                           |
| 文件名                                                       | 20220331_api_design_for_paddle_sparse_sqrt.md<br> |


# 一、概述

## 1、相关背景

sqrt 是一个基础平方根运算操作，目前 Paddle 中有 sparse 的卷积和池化算子，但还没有 sparse 的平方根算子。支持 sparse sqrt 对 Paddle 的稀疏张量生态会有一定的促进作用。

## 2、功能目标

在飞桨中增加 paddle.sparse.sqrt API。

## 3、意义

飞桨将支持 paddle.sparse.sqrt API。

# 二、飞桨现状

目前飞桨还没有稀疏张量的 sqrt 操作。


# 三、业内方案调研

PyTorch 支持 COO 与 CSR 格式的稀疏张量，并且支持许多线性运算操作以及 tensor.add(x)、tensor.sqrt() 等一元、二元操作。

PyTorch 的稀疏张量和 Paddle 一样，不要求值一定是 scalar，而可以是一个 dense tensor。PyTorch 的稀疏一元操作的实现在 aten/src/ATen/native/sparse/SparseUnaryOps.cpp 中，关键代码如下：

```c++
template <typename Ufunc>
Tensor coalesced_unary_ufunc(const Tensor &self, const Ufunc &ufunc) {
  TORCH_INTERNAL_ASSERT(self.is_sparse());
  const auto input = self.coalesce();
  Tensor out_values = ufunc(input.values());
  Tensor result = at::_sparse_coo_tensor_with_dims_and_tensors(
      input.sparse_dim(),
      input.dense_dim(),
      input.sizes(),
      input.indices().clone(),
      out_values,
      input.options().dtype(out_values.scalar_type()));
  result._coalesced_(true);
  return result;
}
```

详细的逻辑是：读取 self 中的非零数据（类型为 dense tensor），传递给 ufunc（ufunc 是一个普通的只支持 dense tensor 的一元函数，如 `at::sqrt`），得到新的非零数据，再使用新的非零数据与原 indices 一起构造一个新的稀疏张量作为输出。

SciPy 也支持 COO 与 CSR 格式的稀疏张量，它的一元操作的实现如下：

```python
# Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op):
        def method(self):
            result = op(self._deduped_data())
            return self._with_data(result, copy=True)

        method.__doc__ = ("Element-wise %s.\n\n"
                          "See `numpy.%s` for more information." % (name, name))
        method.__name__ = name

        return method

    setattr(_data_matrix, name, _create_method(npfunc))
```

`op` 为 `np.sqrt` 等普通的只支持 dense tensor 的一元函数。SciPy 的实现和 PyTorch 类似，取出非零数据后进行操作，使用 `self._with_data` 创建一个和 `self` 有相同的 indices 的稀疏张量。

# 四、对比分析

PyTorch 和 SciPy 的实现是类似的，稀疏张量的一元操作并不复杂。

# 五、设计思路与实现方案

## 命名与参数设计

sparse sqrt 这个稀疏张量上的方法的命名和参数不需要额外设计，在 python/paddle/utils/code_gen/sparse_api.yaml 里新增一项即可。

## 底层OP设计

新增两个 Kernel：

```    cpp
 SparseCooTensor SqrtKernel(const Context& dev_ctx,
 const SparseCooTensor& x,
 SparseCooTensor* out);
```

```cpp
 SparseCsrTensor SqrtKernel(const Context& dev_ctx,
 const SparseCsrTensor& x,
 SparseCsrTensor* out);
```

一元操作的实现较简单，取出 DenseTensor 类型的 non_zero_elements() 后，逐元素进行 sqrt 操作，并创建新的稀疏张量即可。

## API实现方案

API 不需要额外实现，在 python/paddle/utils/code_gen/sparse_api.yaml 里新增一项即可。

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料

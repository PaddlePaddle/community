# paddle.sparse.sum 设计文档

| API名称 | paddle.sparse.sum |
|----------------------------------------------------------|-----------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden"> | 六个骨头 |
| 提交时间<input type="checkbox" class="rowselector hidden"> | 2023-02-22 |
| 版本号 | V1.0 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop |
| 文件名 | 20230222_api_design_for_sparse_sum. md<br> |

# 一、概述
## 1、相关背景
为了提升飞桨 API 丰富度，针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 sum 的计算逻辑，
一共需要新增 2 个 kernel 的前向与反向，其中 COO 格式的 axis 支持任意维度，CSR 格式的 axis 可只支持-1，None，即按行读取。另外当 axis=None 时所有元素相加。

## 3、意义
支持稀疏 tensor 的 sum 操作，丰富基础功能，提升稀疏 tensor 的 API 完整度。

# 二、飞桨现状
目前paddle缺少相关功能实现。

# 三、业内方案调研
## Pytorch
Pytorch中相关实现如下(pytorch/aten/src/ATen/native/sparse/SparseTensorMath.cpp)

```cpp
Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum) {
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  Tensor indices = input._indices();
  Tensor values = input._values();
  IntArrayRef sizes = input.sizes();
  const int64_t sparse_dim = input.sparse_dim();

  auto dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (const auto d : c10::irange(input_dim)) {
    if (dims_to_sum_b[d]) {
      if (d >= sparse_dim) dense_dims_to_sum_v.emplace_back(d + 1 - sparse_dim);
    }
    else {
      dims_to_keep_v.emplace_back(d);
    }
  }
  const int64_t sparse_dims_to_sum_size = dims_to_sum_v.size() - dense_dims_to_sum_v.size();
  const bool sum_all_sparse_dim = (sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (!dense_dims_to_sum_v.empty());

  // new values
  Tensor new_values;
  if (sum_dense_dim) {
    new_values = values.sum(dense_dims_to_sum_v);
  }
  else {
    new_values = values.clone(at::MemoryFormat::Contiguous);
  }

  if (sum_all_sparse_dim) {
    // return a dense tensor if sum over all sparse dims
    new_values = new_values.sum(0);
    return new_values;
  }
  else { // !sum_all_sparse_dim
    // new indices
    Tensor new_indices;
    if (sparse_dims_to_sum_size == 0) {
      new_indices = indices.clone(at::MemoryFormat::Contiguous);
    }
    else {
      new_indices = at::empty({sparse_dim - sparse_dims_to_sum_size, input._nnz()}, indices.options());
      for (auto i: c10::irange(dims_to_keep_v.size())) {
        int64_t d = dims_to_keep_v[i];
        if (d < sparse_dim) new_indices[i].copy_(indices[d]);
        else break;
      }
    }

    // new size
    int64_t new_sparse_dim = new_indices.size(0);
    int64_t new_dense_dim = new_values.dim() - 1; // exclude nnz dim
    std::vector<int64_t> new_sizes;
    new_sizes.reserve(dims_to_keep_v.size());
    for (auto d : dims_to_keep_v) new_sizes.emplace_back(sizes[d]);
    if (sum_all_sparse_dim) new_sizes.emplace(new_sizes.begin(), 1);

    // use coalesce() to do sum reduction
    SparseTensor new_sparse = at::_sparse_coo_tensor_with_dims_and_tensors(new_sparse_dim, new_dense_dim, new_sizes, new_indices, new_values, input.options());
    new_sparse = new_sparse.coalesce();
    return new_sparse;
  }

}
```

## scipy
scipy中相关实现如下(scipy/scipy/sparse/_compressed.py)
```python
def sum(self, axis=None, dtype=None, out=None):
    """Sum the matrix over the given axis.  If the axis is None, sum
    over both rows and columns, returning a scalar.
    """
    # The spmatrix base class already does axis=0 and axis=1 efficiently
    # so we only do the case axis=None here
    if (not hasattr(self, 'blocksize') and
            axis in self._swap(((1, -1), (0, 2)))[0]):
        # faster than multiplication for large minor axis in CSC/CSR
        res_dtype = get_sum_dtype(self.dtype)
        ret = np.zeros(len(self.indptr) - 1, dtype=res_dtype)

        major_index, value = self._minor_reduce(np.add)
        ret[major_index] = value
        ret = self._ascontainer(ret)
        if axis % 2 == 1:
            ret = ret.T

        if out is not None and out.shape != ret.shape:
            raise ValueError('dimensions do not match')

        return ret.sum(axis=(), dtype=dtype, out=out)
    # spmatrix will handle the remaining situations when axis
    # is in {None, -1, 0, 1}
    else:
        return spmatrix.sum(self, axis=axis, dtype=dtype, out=out)

```

## paddle DenseTensor
DenseTensor中的sum被定义为paddle.sum(x, axis=None, dtype=None, keepdim=False, name=None)，
在指定维度上进行求和运算的 Tensor，数据类型和输入数据类型一致。

代码如下
```python
# 例1
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]])
out1 = paddle.sum(x)  # [3.5]
out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

  

# 例2
y = paddle.to_tensor([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]])
out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]
```

但是此处是Dense的，直接使用指针在Sparse中不可行。

# 四、对比分析

为了适配paddle phi库的设计模式，需自行设计实现方式

# 五、方案设计

## 命名与参数设计

在 paddle/phi/kernels/sparse/unary_kernel.h 中，
kernel设计为
```cpp

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out);

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out);

```

在 paddle/phi/kernels/sparse/unary_grad_kernel.h 中， kernel设计为

```cpp
template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx);

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx);
```

并在yaml中新增对应API

sparse_ops.yaml
```yaml
- op : sum
  args : (Tensor x, IntArray axis={}, DataType dtype=DataType::UNDEFINED, bool keepdim=false)
  output : Tensor(out)
  infer_meta :
    func : SumInferMeta
  kernel :
    func : sum_coo{sparse_coo -> sparse_coo},
           sum_csr{sparse_csr -> sparse_csr}
    data_type : x
  backward : sum_grad

```

sparse_backward_ops.yaml
```yaml
- backward_op : sum_grad
  forward : sum(Tensor x, IntArray axis={}, DataType dtype=DataType::UNDEFINED, bool keepdim=false) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, IntArray axis={}, bool keepdim=false)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : sum_coo_grad {sparse_coo -> sparse_coo},
           sum_csr_grad {sparse_csr -> sparse_csr}
  
```

SumInferMeta函数 将优先根据dtype。

## 底层OP设计
对于COO格式和CSR格式`axis=None,keepdim=False`的简单情况，只需要把value值求和，并对相应的位置参数进行置零即可，返回只有一个元素的稀疏Tensor。
对于COO格式和CSR格式`axis=None,keepdim=True`的简单情况，只需要把value值求和，并对相应的位置参数进行修改即可。

对于COO格式的其他情况，主要分为两步，
第一步构建索引（排除掉axis维度）到序号的映射，
得到的结果按序号顺序作为输出结果的索引即可。

对于CSR格式，由于只需要考虑axis=-1的情况，因此对于2维情况，
只需要out_cols全部置零，相邻x_crows有变化的分别改为变化量为1，
无变化保持无变化，而value是相应的求和，大致逻辑代码如下
```cpp
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out) {
  size_t n_dim = axis.size();
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const T* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  if (n_dim == 0) {
    out_dims = make_ddim({1, 1});
    out_crows = Empty<int64_t, Context>(dev_ctx, {2});  // crows = [0, 1]
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;
    out_crows_data[0] = 1;

    out_cols = Empty<int64_t, Context>(dev_ctx, {1});  // crows = [0]
    auto* out_cols_data = out_cols.data<int64_t>();
    out_cols_data[0] = 0;

    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_dims = make_ddim({x.dims()[0], 1});
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;

    std::vector<T> out_data;
    for (int i = 0; i < x.dims()[0]; ++i) {
      if (x_crows_data[i] != x_crows_data[i + 1]) {
        int sum_value = 0;
        for (auto j = x_crows_data[i]; j < x_crows_data[i + 1]; ++j) {
          sum_value += x_values_data[j];
        }
        out_crows_data[i + 1] = out_crows_data[i] + 1;
        out_data.push_back(sum_value);
      } else {
        out_crows_data[i + 1] = out_crows_data[i];
      }
    }

    out_cols =
        Empty<int64_t, Context>(dev_ctx, {static_cast<int>(out_data.size())});
    out_values =
        Empty<T, Context>(dev_ctx, {static_cast<int>(out_data.size())});
    auto* out_cols_data = out_cols.data<int64_t>();
    T* out_values_data = out_values.data<T>();
    for (size_t i = 0; i < out_data.size(); ++i) {
      out_cols_data[i] = 0;
      out_values_data[i] = out_data[i];
    }
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}
```
对于3维情况，按层考虑即可。

对于反向传播，dx的coo和csr的索引（indices、crows、cols）均与x保持不变，
value则取决于dout的相应位置值。

## API实现方案

对于SparseCsrTensor和SparseCooTensor有相同的API，

均只需要给定输入张量和维度转换目标。

具体的API为`paddle.sparse.sum(x, axis=None, dtype=None, keepdim=False)`

- x: 输入张量
- axis: 求和的维度，例如-1表示最后一个维度求和。
- dtype: 输出张量的类型，
- keepdim: 是否保持输入和输出维度不变，
例如$[5, 5]$的张量输入对第一个维度求和，若保持维度不变则输出为$[1, 5]$，否则为$[5]$。

# 六、测试和验收的考量

测试考虑的case如下：

- 正确性
- csr对2维和3维`axis=-1`及None参数测试
- coo对1维、2维、3维、6维和10维不同`axis`参数测试
- coo、csr分别对dtype缺省和特定值测试
- 分别对每个测试样本分成keepdim为真或假

具体样例如下

```python
class TestSum(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, dims, keepdim, format, dtype=None):
        mask = paddle.randint(0, 2, x_shape).astype("float32")
        # "+ 1" to make sure that all zero elements in "origin_x" is caused by multiplying by "mask",
        # or the backward checks may fail.
        origin_x = (paddle.rand(x_shape, dtype='float32') + 1) * mask
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_out = paddle.sum(dense_x, dims, keepdim=keepdim, dtype=dtype)
        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.sum(sp_x, dims, keepdim=keepdim, dtype=dtype)

        np.testing.assert_allclose(
            sp_out.to_dense().numpy(), dense_out.numpy(), rtol=1e-05
        )
        dense_out.backward()
        sp_out.backward()
        np.testing.assert_allclose(
            sp_x.grad.to_dense().numpy(),
            (dense_x.grad * mask).numpy(),
            rtol=1e-05,
        )

    def test_sum_1d(self):
        self.check_result([5], None, False, 'coo')
        self.check_result([5], None, True, 'coo')
        self.check_result([5], 0, True, 'coo')
        self.check_result([5], 0, False, 'coo')

    def test_sum_2d(self):
        self.check_result([2, 5], None, False, 'coo')
        self.check_result([2, 5], None, True, 'coo')
        self.check_result([2, 5], 0, True, 'coo')
        self.check_result([2, 5], 0, False, 'coo')
        self.check_result([2, 5], 1, False, 'coo')
        self.check_result([2, 5], None, True, 'csr')
        self.check_result([2, 5], -1, True, 'csr')
        self.check_result([2, 5], 0, False, 'coo', dtype="int32")
        self.check_result([2, 5], -1, True, 'csr', dtype="int32")

    def test_sum_3d(self):
        self.check_result([6, 2, 3], -1, True, 'csr')
        for i in [0, 1, -2, None]:
            self.check_result([6, 2, 3], i, False, 'coo')
            self.check_result([6, 2, 3], i, True, 'coo')

    def test_sum_nd(self):
        for i in range(6):
            self.check_result([8, 3, 4, 4, 5, 3], i, False, 'coo')
            self.check_result([8, 3, 4, 4, 5, 3], i, True, 'coo')
            # Randint now only supports access to dimension 0 to 9.
            self.check_result([2, 3, 4, 2, 3, 4, 2, 3, 4], i, False, 'coo')

```


# 七、可行性分析及规划排期

方案主要自行实现核心算法
预计3.10号前完成cpu部分的实现和测试
预计3.10号前完成gpu部分的实现和测试
预计4.1号前完成各种参数的实现和测试
预计4.15号前完成文档

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
 
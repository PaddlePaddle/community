# paddle.sparse.any 设计文档

| API名称                                                      | paddle.sparse.any                           |
|------------------------------------------------------------|---------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">     | zrr1999                                     |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2023-04-05                                  |
| 版本号                                                        | V1.0                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">   | develop                                     |
| 文件名                                                        | 20230405_api_design_for_sparse_any.md<br>   |

# 一、概述
## 1、相关背景
为了提升飞桨 API 丰富度，针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 any 的计算逻辑，
any 的计算逻辑是元素中任意一个为真结果即为真。
一共需要新增 2 个 kernel 的前向，其中 COO 格式的 axis 支持任意维度，CSR 格式的 axis 可只支持-1，即按行读取。
另外当 axis=None 时所有元素进行 any 运算。

## 3、意义
支持稀疏 tensor 的 any 操作，丰富基础功能，提升稀疏 tensor 的 API 完整度。

# 二、飞桨现状
目前paddle缺少相关功能实现。

# 三、业内方案调研
## Pytorch
Pytorch 中相关实现如下(pytorch/aten/src/ATen/native/ReduceOps.cpp)，
PyTorch 只支持COO格式的稀疏张量，且与普通张量的实现放在了一起。

```cpp
template <int identity, typename Stub>
inline void allany_impl(
    const Tensor& self,
    const Tensor& result,
    IntArrayRef dims,
    bool keepdim,
    Stub& stub) {
  if (self.numel() == 0) {
    result.fill_(identity);
  } else if (self.numel() == 1) {
    result.copy_(self.view_as(result).to(at::kBool));
  } else {
    auto iter = get_allany_iter(self, result, dims, keepdim);
    stub(iter.device_type(), iter);
  }
}

TORCH_IMPL_FUNC(all_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<1>(self, result, dim, keepdim, and_stub);
}

TORCH_IMPL_FUNC(all_all_out)(const Tensor& self, const Tensor& result) {
  allany_impl<1>(self, result, {}, false, and_stub);
}

TORCH_IMPL_FUNC(any_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<0>(self, result, dim, keepdim, or_stub);
}
```

## scipy
scipy.sparse库中没有any()函数。但是，可以使用numpy库中的any()函数在稀疏矩阵中执行相同的操作。

## paddle DenseTensor
DenseTensor中的any被定义为paddle.any(x, axis=None, dtype=None, keepdim=False, name=None)，
在指定维度上进行进行逻辑或运算的 Tensor，数据类型和输入数据类型一致。
代码如下
```python
import paddle

x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
x = paddle.assign(x)
print(x)
x = paddle.cast(x, 'bool')
# x is a bool Tensor with following elements:
#    [[True, False]
#     [True, True]]

# out1 should be [True]
out1 = paddle.any(x)  # [True]
print(out1)

# out2 should be [True, True]
out2 = paddle.any(x, axis=0)  # [True, True]
print(out2)

# keepdim=False, out3 should be [True, True], out.shape should be (2,)
out3 = paddle.any(x, axis=-1)  # [True, True]
print(out3)

# keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
out4 = paddle.any(x, axis=1, keepdim=True)  # [[True], [True]]
print(out4)
```

但是此处是Dense的，直接使用指针在Sparse中不可行。

# 四、对比分析

为了适配paddle phi库的设计模式，需自行设计实现方式

# 五、方案设计

## 命名与参数设计

在 paddle/phi/kernels/sparse/unary_kernel.h 中， kernel设计为

```cpp

template <typename T, typename Context>
void AnyCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out);

template <typename T, typename Context>
void AnyCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out);

```

在 paddle/phi/kernels/sparse/unary_grad_kernel.h 中， kernel设计为

```cpp
template <typename T, typename Context>
void AnyCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx);

template <typename T, typename Context>
void AnyCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx);
```

并在yaml中新增对应API

sparse_ops.yaml
```yaml
- op : any
  args : (Tensor x, IntArray axis={}, bool keepdim=false)
  output : Tensor(out)
  infer_meta :
    func : AnyInferMeta
  kernel :
    func : any_coo{sparse_coo -> sparse_coo},
           any_csr{sparse_csr -> sparse_csr}
    data_type : x
  backward : any_grad

```

sparse_backward_ops.yaml
```yaml
- backward_op : any_grad
  forward : any(Tensor x, IntArray axis={}, bool keepdim=false) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, IntArray axis={}, bool keepdim=false)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : any_coo_grad {sparse_coo -> sparse_coo},
           any_csr_grad {sparse_csr -> sparse_csr}
  
```

相应的InferMeta函数可以复用稠密矩阵的函数。

## 底层OP设计
对于axis=None的简单情况，只需要把value值进行逻辑或运算，并对相应的位置参数进行修改即可。

对于COO格式的其他情况，主要分为两步，
第一步构建索引（排除掉axis维度）到序号的映射，
得到的结果按序号顺序作为输出结果的索引即可。

对于CSR格式，由于只需要考虑axis=-1的情况，因此对于2维情况，
只需要out_cols全部置零，相邻x_crows有变化的分别改为变化量为1，
无变化保持无变化，而value是相应的逻辑或运算，大致逻辑代码如下
```cpp
void AnyCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
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

    out_values = phi::Any<T>(dev_ctx, x.values(), {}, true);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of AnyCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_dims = make_ddim({x.dims()[0], 1});
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;

    std::vector<T> out_data;
    for (int i = 0; i < x.dims()[0]; ++i) {
      if (x_crows_data[i] != x_crows_data[i + 1]) {
        bool any_value = False;
        for (auto j = x_crows_data[i]; j < x_crows_data[i + 1]; ++j) {
          any_value ||= x_values_data[j];
        }
        out_crows_data[i + 1] = out_crows_data[i] + 1;
        out_data.push_back(any_value);
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

具体的API为`paddle.sparse.any(x, axis=None, keepdim=False)`

- x: 输入张量
- axis: 进行逻辑或运算的维度，例如-1表示最后一个维度进行逻辑或运算。
- keepdim: 是否保持输入和输出维度不变，
例如$[5, 5]$的张量输入对第一个维度进行逻辑或运算，若保持维度不变则输出为$[1, 5]$，否则为$[5]$。

# 六、测试和验收的考量

测试考虑的case如下：

- 正确性
- csr对2维和3维不同`axis`参数测试
- coo对1维、2维、3维、6维和10维不同`axis`参数测试
- 分别对每个测试样本分成keepdim为真或假

具体样例如下

```python
class TestAny(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, dims, keepdim, format):
        mask = paddle.randint(0, 2, x_shape).astype("float32")
        # "+ 1" to make sure that all zero elements in "origin_x" is caused by multiplying by "mask",
        # or the backward checks may fail.
        origin_x = (paddle.rand(x_shape, dtype='float32') + 1) * mask
        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_out = paddle.any(dense_x, dims, keepdim=keepdim)
        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.sparse.any(sp_x, dims, keepdim=keepdim)

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

    def test_any_1d(self):
        self.check_result([5], None, False, 'coo')
        self.check_result([5], None, True, 'coo')
        self.check_result([5], 0, True, 'coo')
        self.check_result([5], 0, False, 'coo')

    def test_any_2d(self):
        self.check_result([2, 5], None, False, 'coo')
        self.check_result([2, 5], None, True, 'coo')
        self.check_result([2, 5], 0, True, 'coo')
        self.check_result([2, 5], 0, False, 'coo')
        self.check_result([2, 5], 1, False, 'coo')
        self.check_result([2, 5], None, True, 'csr')
        self.check_result([2, 5], -1, True, 'csr')

    def test_any_3d(self):
        self.check_result([6, 2, 3], -1, True, 'csr')
        for i in [0, 1, -2, None]:
            self.check_result([6, 2, 3], i, False, 'coo')
            self.check_result([6, 2, 3], i, True, 'coo')

    def test_any_nd(self):
        for i in range(6):
            self.check_result([8, 3, 4, 4, 5, 3], i, False, 'coo')
            self.check_result([8, 3, 4, 4, 5, 3], i, True, 'coo')
            # Randint now only supports access to dimension 0 to 9.
            self.check_result([2, 3, 4, 2, 3, 4, 2, 3, 4], i, False, 'coo')

class TestSparseAnyStatic(unittest.TestCase):
    def check_result_coo(self, x_shape, dims, keepdim):
        mask = paddle.randint(0, 2, x_shape)
        origin_data = (paddle.rand(x_shape, dtype='float32') + 1) * mask
        sparse_data = origin_data.detach().to_sparse_coo(
            sparse_dim=len(x_shape)
        )
        indices_data = sparse_data.indices()
        values_data = sparse_data.values()

        dense_x = origin_data
        dense_out = paddle.any(dense_x, dims, keepdim=keepdim)

        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            indices = paddle.static.data(
                name='indices',
                shape=indices_data.shape,
                dtype=indices_data.dtype,
            )
            values = paddle.static.data(
                name='values', shape=values_data.shape, dtype=values_data.dtype
            )
            sp_x = paddle.sparse.sparse_coo_tensor(
                indices,
                values,
                shape=origin_data.shape,
                dtype=origin_data.dtype,
            )
            sp_out = paddle.sparse.any(sp_x, dims, keepdim=keepdim)
            sp_dense_out = sp_out.to_dense()

            sparse_exe = paddle.static.Executor()
            sparse_fetch = sparse_exe.run(
                feed={
                    'indices': indices_data.numpy(),
                    "values": values_data.numpy(),
                },
                fetch_list=[sp_dense_out],
                return_numpy=True,
            )

            np.testing.assert_allclose(
                dense_out.numpy(), sparse_fetch[0], rtol=1e-5
            )
        paddle.disable_static()

    def test_any_1d(self):
        self.check_result_coo([5], None, False)
        self.check_result_coo([5], None, True)
        self.check_result_coo([5], 0, True)
        self.check_result_coo([5], 0, False)

        self.check_result_coo([2, 5], None, False)
        self.check_result_coo([2, 5], None, True)
        self.check_result_coo([2, 5], 1, True)
        self.check_result_coo([2, 5], 0, True)
        self.check_result_coo([2, 5], 1, False)
        self.check_result_coo([2, 5], 0, False)

        for i in [0, 1, -2, None]:
            self.check_result_coo([6, 2, 3], i, False)
            self.check_result_coo([6, 2, 3], i, True)

        for i in range(6):
            self.check_result_coo([8, 3, 4, 4, 5, 3], i, False)
            self.check_result_coo([8, 3, 4, 4, 5, 3], i, True)
            # Randint now only supports access to dimension 0 to 9.
            self.check_result_coo([2, 3, 4, 2, 3, 4, 2, 3, 4], i, False)
```


# 七、可行性分析及规划排期

方案主要自行实现核心算法
预计5.10号前完成cpu部分的实现和测试
预计5.10号前完成gpu部分的实现和测试
预计5.15号前完成各种参数的实现和测试
预计5.20号前完成文档

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
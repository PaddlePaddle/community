# paddle.incubate.sparse.transpose 设计文档

| API名称                                                    | paddle.incubate.sparse.transpose                |
|----------------------------------------------------------|-----------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | 六个骨头                                       |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-07                                    |
| 版本号                                                      | V1.0                                          |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                       |
| 文件名                                                      | 20220907_api_design_for_sparse_transpose.md<br> |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 transpose 的计算逻辑，
一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

## 3、意义

支持稀疏tensor的transpose操作，丰富基础功能，提升稀疏tensor的API完整度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中相关实现如下

```c
static inline Tensor & sparse_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  int64_t nsparse_dim = self.sparse_dim();
  TORCH_CHECK(dim0 < nsparse_dim && dim1 < nsparse_dim,
           "sparse transpose: transposed dimensions must be sparse ",
           "Got sparse_dim: ", nsparse_dim, ", d0: ", dim0, ", d1: ", dim1);

  if (self._indices().numel() == 0 && self._values().numel() == 0) {
    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self.sparse_dim(), self.dense_dim(), sizes);
  } else {
    auto indices = self._indices();
    auto row0 = indices.select(0, dim0);
    auto row1 = indices.select(0, dim1);

    // swap row0 and row1
    auto tmp = at::zeros_like(row0, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    tmp.copy_(row0);
    row0.copy_(row1);
    row1.copy_(tmp);

    self._coalesced_(false);

    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self._indices().size(0), self._values().dim() - 1, sizes);
  }
  return self;
}
```
## scipy
scipy中转换为csr再进行transpose
```python
def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse matrix.
        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse matrix
            being used.
        Returns
        -------
        p : `self` with the dimensions reversed.
        See Also
        --------
        numpy.matrix.transpose : NumPy's implementation of 'transpose'
                                 for matrices
        """
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)
```
csr transpose实现如下
```python

def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape
        return self._csc_container((self.data, self.indices,
                                    self.indptr), shape=(N, M), copy=copy)

```
## paddle DenseTensor
参数dims在DenseTensor中被表达为perm，其长度与输入张量的维度必须相等，
返回多维张量的第i维对应输入Tensor的perm[i]维。。

代码如下
```python
x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
    [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
shape(x) =  [2,3,4]

# 例0
perm0 = [1,0,2]
y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
          [[ 5  6  7  8]  [17 18 19 20]]
          [[ 9 10 11 12]  [21 22 23 24]]]
shape(y_perm0) = [3,2,4]

# 例1
perm1 = [2,1,0]
y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
          [[ 2 14] [ 6 18] [10 22]]
          [[ 3 15]  [ 7 19]  [11 23]]
          [[ 4 16]  [ 8 20]  [12 24]]]
shape(y_perm1) = [4,3,2]
```
但是此处是Dense的，直接使用指针在Sparse中不可行
# 四、对比分析
为了适配paddle phi库的设计模式，需自行设计实现方式
# 五、方案设计
## 命名与参数设计
在 paddle/phi/kernels/sparse/unary_kernel.h 中， kernel设计为
```cpp
template <typename T, typename Context>
void TransposeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int>& perm,
                        SparseCooTensor* out);
template <typename T, typename Context>
void TransposeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const std::vector<int>& perm,
                        SparseCsrTensor* out);
```
在 paddle/phi/kernels/sparse/unary_grad_kernel.h 中， kernel设计为
```
template <typename T, typename Context>
void TransposeCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& dout,
                            const std::vector<int>& perm,
                            SparseCooTensor* dx);
template <typename T, typename Context>
void TransposeCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& dout,
                            const std::vector<int>& perm,
                            SparseCsrTensor* dx);
```
并在yaml中新增对应API
```yaml
- op : transpose
  args : (Tensor x, int[] perm)
  output : Tensor(out)
  infer_meta :
    func : TransposeInferMeta
  kernel :
    func : transpose_coo{sparse_coo -> sparse_coo},
           transpose_csr{sparse_csr -> sparse_csr}
    layout : x
  backward : transpose_grad

```
```yaml
- backward_op : transpose_grad
  forward : transpose(Tensor x, int[] perm) -> Tensor(out)
  args : (Tensor out, Tensor out_grad)
  output : Tensor(x_grad)
  infer_meta :
    func : TransposeGradInferMeta
    param : [out_grad, dims]
  kernel :
    func : transpose_coo_grad {sparse_coo, sparse_coo -> sparse_coo},
           transpose_csr_grad {sparse_csr, sparse_csr -> sparse_csr}

```
以及实现相应的InferMeta函数。
## 底层OP设计
对于Coo格式，主要分为两步，第一步操作indices，通过遍历每一行，
按照指定顺序复制给输出值，第二步使用DDim::transpose改变dims值。

对于Csr格式，通过分类讨论的方式，分别实现2维和3维的功能。
对于2维情况只需要确定两个维度是否切换，对于不切换直接返回输入张量的副本，
若转置，需要后文介绍的方法进行转置；3维情况较为复杂，对于`perm`输入[0, 2, 1]，
可直接参考2维情况，对于`perm`输入[1, 0, 2]，也在后文介绍，对于其他情况，
均可以通过组合前两种情况得到，例如[1, 2, 0]可视为[1, 0, 2]和[0, 2, 1]的组合。
梯度可以转化为相应的transpose算子实现，cuda实现只需要对相应的循环进行替换和不断优化即可。

2维情况
```c++
for (int i = 0; i < out_dims[0]; ++i) {
  out_crows_data[i] = 0;
}
for (int i = 0; i < x_nnz; ++i) {
  int j = x_cols_data[i];
  out_crows_data[j + 1]++;
}
out_crows_data[out_dims[0]] = x_nnz;
for (int i = 1; i < out_dims[0]; ++i) {
  out_crows_data[i] += out_crows_data[i - 1];
}
// compute out_cols_data and out_values_data by out_crows_data and x
std::unordered_map<int64_t, int> cols_offset;
for (int i = 0; i < x.dims()[0]; ++i) {
  int64_t start = x_crows_data[i];
  int64_t end = x_crows_data[i + 1];
  for (int64_t j = start; j < end; ++j) {
    int64_t x_cols_j = x_cols_data[j];
    int64_t jjj = out_crows_data[x_cols_j];
    if (cols_offset.count(jjj)) {
      cols_offset[jjj]++;
    } else {
      cols_offset[jjj] = 0;
    }
    int64_t jjj_offset = jjj + cols_offset[jjj];
    out_cols_data[jjj_offset] = i;
    out_values_data[jjj_offset] = x_values_data[j];
  }
}
```
`perm`输入[1, 0, 2]的情况
```c++
// k 可视为输出的第一个维度索引
for (int i = 0; i < out_n_rows; ++i) {
  out_crows_data[i] = 0;
}
int x_cols_offset = 0;
int out_cols_index = 0;
for (int i = 0; i < x.dims()[0]; ++i) {
  int x_crows_index = i * (x_n_rows + 1);
  int start = x_crows_data[x_crows_index + k];
  int end = x_crows_data[x_crows_index + 1 + k];
  out_crows_data[i + 1] = end - start;
  for (int j = start; j < end; ++j) {
    out_cols_data[out_cols_index] = x_cols_data[x_cols_offset + j];
    out_values_data[out_cols_index] = x_values_data[x_cols_offset + j];
    out_cols_index++;
  }
  x_cols_offset += x_crows_data[x_crows_index + x_n_rows];
}
for (int i = 1; i <= out_n_rows; ++i) {
  out_crows_data[i] += out_crows_data[i - 1];
}
```
## API实现方案
对于SparseCsrTensor和SparseCooTensor有相同的API，
均只需要给定输入张量和维度转换目标。
具体的API为`paddle.incubate.sparse.transpose(x, perm)`
- x: 输入张量
- perm: 变换的维度，例如[0, 2, 1]表示对后两个维度进行对换，必须保证与输入张量尺寸的长度相等。

# 六、测试和验收的考量
测试考虑的case如下：
- 正确性
- csr对2维和3维不同`perm`参数测试
- coo对2维、3维不同`perm`参数以及6维和10维测试

具体样例如下
```python
class TestTranspose(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, perm, format):
        with _test_eager_guard():
            mask = paddle.randint(0, 2, x_shape).astype("float32")
            origin_x = paddle.rand(x_shape, dtype='float32') * mask
            dense_x = origin_x.detach()
            dense_x.stop_gradient = False
            dense_out = paddle.transpose(dense_x, perm)

            if format == "coo":
                sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
            else:
                sp_x = origin_x.detach().to_sparse_csr()
            sp_x.stop_gradient = False
            sp_out = paddle.incubate.sparse.transpose(sp_x, perm)
            np.testing.assert_allclose(sp_out.to_dense().numpy(),
                                       dense_out.numpy(),
                                       rtol=1e-05)
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                       (dense_x.grad * mask).numpy(),
                                       rtol=1e-05)

    def test_transpose_2d(self):
        self.check_result([2, 5], [0, 1], 'coo')
        self.check_result([2, 5], [0, 1], 'csr')
        self.check_result([2, 5], [1, 0], 'coo')
        self.check_result([2, 5], [1, 0], 'csr')

    def test_transpose_3d(self):
        self.check_result([6, 2, 3], [0, 1, 2], 'coo')
        self.check_result([6, 2, 3], [0, 1, 2], 'csr')
        self.check_result([6, 2, 3], [0, 2, 1], 'coo')
        self.check_result([6, 2, 3], [0, 2, 1], 'csr')
        self.check_result([6, 2, 3], [1, 0, 2], 'coo')
        self.check_result([6, 2, 3], [1, 0, 2], 'csr')
        self.check_result([6, 2, 3], [2, 0, 1], 'coo')
        self.check_result([6, 2, 3], [2, 0, 1], 'csr')
        self.check_result([6, 2, 3], [2, 1, 0], 'coo')
        self.check_result([6, 2, 3], [2, 1, 0], 'csr')
        self.check_result([6, 2, 3], [1, 2, 0], 'coo')
        self.check_result([6, 2, 3], [1, 2, 0], 'csr')

    @unittest.skipIf(paddle.is_compiled_with_cuda(),
                     "cuda randint not supported")
    def test_transpose_nd(self):
        self.check_result([8, 3, 4, 4, 5, 3], [5, 3, 4, 1, 0, 2], 'coo')
        # Randint now only supports access to dimension 0 to 9.
        self.check_result([i % 3 + 2 for i in range(9)],
                          [(i + 2) % 9 for i in range(9)], 'coo')

```

# 七、可行性分析及规划排期
方案主要自行实现核心算法
# 八、影响面
为独立新增op，对其他模块没有影响
# 名词解释
无
# 附件及参考资料
无
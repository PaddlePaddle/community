# paddle.sparse.slice 设计文档


| API名称      | paddle.sparse.slice                     |
|-------------|-----------------------------------------|
| 提交作者     | ScottWong98                             |
| 提交时间     | 2023-02-25                              |
| 版本号       | V1.0.0                                  |
| 依赖飞桨版本  | develop                                 |
| 文件名       | 20230225_api_design_for_sparse_slice.md |

# 一、概述
## 1、相关背景

简单来说，稀疏 Tensor 是元素大部分为零的矩阵，在实际求解任务时经常出现大规模的稀疏 Tensor。由于其自身的稀疏性，为了节省存储空间，通常会修改稀疏 Tensor 的存储结构。目前比较普遍的存储结构为 COO 和 CSR。

Paddle 目前已经实现了 COO 和 CSR 格式的稀疏 Tensor 的构建以及一些算子操作，然而目前还没有支持对其的 slice 操作，而 slice 操作在实际中是有应用价值的，因此在 Paddle 中集成该功能是有必要的。

## 2、功能目标

为 Paddle 新增 paddle.sparse.slice 稀疏 API。针对 Paddle 的两种稀疏 Tensor 格式 COO 和 CSR，都需新增 slice 的计算逻辑。一共需要新增 2 个 kernel 的前向与反向。动静态图都需要支持。

其中 COO 的 kernel 需要支持任意维度的稀疏 Tensor，CSR 的 kernel 需要支持 2D/3D 的稀疏 Tensor。

## 3、意义

支持稀疏 Tensor 的 slice 操作，丰富基础功能，提升稀疏 Tensor 的 API 完整度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

针对 PyTorch，TensorFlow 和 SciPy 三种框架对该功能进行了调研，具体结果如下。

## PyTorch
PyTorch 目前还不支持对稀疏 Tensor 的 slice 功能，参考 [PyTorch 论坛上的回答](https://discuss.pytorch.org/t/column-row-slicing-a-torch-sparse-tensor/19130/2)。

## TensorFlow

TensorFlow 只支持 COO 格式的 slice 功能。详情可参考官方文档（[tf.sparse.slice](https://www.tensorflow.org/api_docs/python/tf/sparse/slice)）。

具体核心实现代码如下所示（截取自 [tensorflow/core/util/sparse/sparse_tensor.h](https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/core/util/sparse/sparse_tensor.h#L580) 文件）：

```cpp
template <typename T>
inline StatusOr<SparseTensor> SparseTensor::Slice(
    const SparseTensor& input_tensor, const gtl::ArraySlice<int64_t> start,
    const gtl::ArraySlice<int64_t> size) {
  TensorShape output_shape(input_tensor.shape());

  const int dims = input_tensor.dims();
  for (int dim = 0; dim < dims; dim++) {
    // Determine the size of the result; if the selected slice goes beyond the
    // input boundary, the result will correspond to the size of the overlap
    // between the input and the selected slice.
    const int64_t input_size = output_shape.dim_size(dim);
    const int64_t start_index = start[dim];
    const int64_t slice_size = size[dim];

    if (start_index < input_size - slice_size) {
      // The entire selection is within input boundaries.
      TF_RETURN_IF_ERROR(output_shape.SetDimWithStatus(dim, slice_size));
    } else if (start_index < input_size) {
      // The selection starts within input boundaries, but goes beyond them.
      TF_RETURN_IF_ERROR(
          output_shape.SetDimWithStatus(dim, input_size - start_index));
    } else {
      // The selection is entirely out of input boundaries.
      TF_RETURN_IF_ERROR(output_shape.SetDimWithStatus(dim, 0));
    }
  }

  auto input_indices_t = input_tensor.indices().matrix<int64_t>();
  auto input_values_t = input_tensor.values().vec<T>();

  // Find the number of indices that fall inside start and size.
  int count = 0;
  for (int i = 0; i < input_tensor.indices().dim_size(0); i++) {
    // The following will check to see if an input is within the
    // range specified by start and size.
    // The for loop below iterates through all dimensions. In case
    // the index falls outside of the start and size at any dimension,
    // it will be considered as a "no hit" (hit = false). In this
    // case, it will not be counted as the index that fall inside
    // the range specified by start and size.
    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) &&
            input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }
    count++;
  }

  Tensor output_values(DataTypeToEnum<T>::v(), TensorShape({count}));
  Tensor output_indices(DT_INT64, TensorShape({count, dims}));

  auto output_values_t = output_values.vec<T>();
  auto output_indices_t = output_indices.matrix<int64_t>();

  // Obtain the output indices that fall inside start and size.
  int index = 0;
  for (int i = 0; i < input_tensor.indices().dim_size(0) && index < count;
       i++) {
    // The logic here is similar as the above except that the above
    // only count the number of indices while here we actually generate
    // the output.
    bool hit = true;
    for (int dim = 0; dim < dims; dim++) {
      if (!(start[dim] <= input_indices_t(i, dim) &&
            input_indices_t(i, dim) < start[dim] + size[dim])) {
        hit = false;
        break;
      }
    }
    if (!hit) {
      continue;
    }
    output_values_t(index) = input_values_t(i);
    for (int dim = 0; dim < dims; dim++) {
      output_indices_t(index, dim) = input_indices_t(i, dim) - start[dim];
    }
    index++;
  }

  return SparseTensor(output_indices, output_values, output_shape);
}
```

## SciPy

SciPy 只支持对 CSR 格式的 slice 操作。SciPy 并没有提供对 slice 操作的文档说明，但经过实践，发现与 Numpy 中的 slice 操作形式一样。

SciPy 中对 slice 操作的具体核心实现代码如下所示 (截取自 [scipy/sparse/sparsetools/csr.h](https://github.com/scipy/scipy/blob/v1.10.1/scipy/sparse/sparsetools/csr.h#L1181) 文件)：
```c++
template<class I, class T>
void get_csr_submatrix(const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                       const T Ax[],
                       const I ir0,
                       const I ir1,
                       const I ic0,
                       const I ic1,
                       std::vector<I>* Bp,
                       std::vector<I>* Bj,
                       std::vector<T>* Bx)
{
    I new_n_row = ir1 - ir0;
    //I new_n_col = ic1 - ic0;  //currently unused
    I new_nnz = 0;
    I kk = 0;

    // Count nonzeros total/per row.
    for(I i = 0; i < new_n_row; i++){
        I row_start = Ap[ir0+i];
        I row_end   = Ap[ir0+i+1];

        for(I jj = row_start; jj < row_end; jj++){
            if ((Aj[jj] >= ic0) && (Aj[jj] < ic1)) {
                new_nnz++;
            }
        }
    }

    // Allocate.
    Bp->resize(new_n_row+1);
    Bj->resize(new_nnz);
    Bx->resize(new_nnz);

    // Assign.
    (*Bp)[0] = 0;
    for(I i = 0; i < new_n_row; i++){
        I row_start = Ap[ir0+i];
        I row_end   = Ap[ir0+i+1];

        for(I jj = row_start; jj < row_end; jj++){
            if ((Aj[jj] >= ic0) && (Aj[jj] < ic1)) {
                (*Bj)[kk] = Aj[jj] - ic0;
                (*Bx)[kk] = Ax[jj];
                kk++;
            }
        }
        (*Bp)[i+1] = kk;
    }
}
```

# 四、对比分析

由于 PyTorch 并没有支持稀疏 Tensor 的 slice 操作，故我们只对 TensorFlow 和 SciPy 进行分析。

TensorFlow
- 优点：实现了 COO 格式下对任意维度 slice 的操作
- 缺点：仅支持 COO 格式

SciPy
- 优点：实现了 CSR 格式下 slice 的操作
- 缺点：
  - 仅提供 CSR 格式的 API，对于 COO 格式的 slice 操作，只能转换到 CSR 格式进行实现。
  - 只支持 2D 稀疏 Tensor 的 slice 操作

因此，我们可以在 TensorFlow 和 SciPy 的实现逻辑之上进行相应的改动，来实现我们所设置的功能目标。
# 五、设计思路与实现方案

## 命名与参数设计

仿照 `DenseTensor` 中 slice kernel 的设计，在 `paddle/phi/kernels/sparse/cpu/slice_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/slice_kernel.cu` 中，前向 kernel 的设计为：
```c++
template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCooTensor* out);
```
```c++
template <typename T, typename Context>
void SliceCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCsrTensor* out);
```

在 `paddle/phi/kernels/sparse/cpu/slice_grad_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/slice_grad_kernel.cu` 中，反向 kernel 的设计为：
```c++
template <typename T, typename Context>
void SliceCooGradKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCooTensor* x_grad);
```
```c++
template <typename T, typename Context>
void SliceCsrGradKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCsrTensor* x_grad);
```

在 `paddle/phi/api/yaml/sparse_ops.yaml` 中新增对应 API：
```yaml
- op : slice
  args : (Tensor x, IntArray axes, IntArray starts, IntArray ends)
  output : Tensor(out)
  infer_meta :
    func : UnchangedInferMeta
    param: [x]
  kernel :
    func : slice_coo{sparse_coo -> sparse_coo},
           slice_csr{sparse_csr -> sparse_csr}
    layout: x
  backward : slice_grad
```

在 `paddle/phi/api/yaml/sparse_backward.yaml` 中新增对应 API：
```yaml
- backward_op : slice_grad
  forward : slice (Tensor x, IntArray axes, IntArray starts, IntArray ends) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, IntArray axes, IntArray starts, IntArray ends)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : slice_coo_grad{sparse_coo, sparse_coo -> sparse_coo},
           slice_csr_grad{sparse_csr, sparse_csr -> sparse_csr}
```
## 底层OP设计

对于 COO 格式的 slice 操作，可以参考 TensorFlow 的方法，遍历每个非零元素，判断其位置在各维度上是否在 slice 的范围内。

对于 CSR 格式的 slice 操作，可以在 SciPy 的基础上添加对 3D 稀疏 Tensor 的 slice 操作。
- 对于 2D 稀疏 Tensor，处理逻辑与 SciPy 相似
- 对于 3D 稀疏 Tensor，可以先对第 0 维进行 slice，第 1 维和第 2 维的处理与 2D 稀疏 Tensor 的处理逻辑类似。

## API实现方案

预期 Paddle 调用 slice API 的形式为：
```python
paddle.sparse.slice(x, axes, starts, ends)
```
- **x** (Tensor) - 输入的稀疏 Tensor，支持 COO 和 CSR 格式
- **axes** (list|tuple|Tensor) - 需要进行 slice 操作的维度，如果是 CSR 格式的稀疏 Tensor，确保长度为 2 或 3
- **starts** (list|tuple|Tensor) - 各维度上 slice 的起始位置，如果是 CSR 格式的稀疏 Tensor，确保长度为 2 或 3
- **ends** (list|tule|Tensor) - 各维度上 slice 的结束位置，如果是 CSR 格式的稀疏 Tensor，确保长度为 2 或 3

我们会首先检查 **axes**, **starts** 与 **ends** 的合法性，再进行对应的 slice 操作。

# 六、测试和验收的考量

测试考虑的 case 以及验收标准如下：

| case | 验收标准|
|------|-------|
|axes, starts 和 ends 长度对比 | 对长度不相等的情况能进行报错，相等的情况能返回正确结果|
|axes, starts 和 ends 对边界的处理 | 对超出边界的情况能进行报错，未超出边界的情况能返回正确结果|
|axes, starts 和 ends 对负数的处理 | 能返回正确结果|
|不同 shape, axes, starts 和 ends 下结果的正确性 | 能返回正确结果|

# 七、可行性分析和排期规划

方案主要自行实现核心算法，可行。具体规划为：

- 阶段一：实现 cpu 上的 API 功能开发，并通过测试
- 阶段二：实现 gpu 上的 API 功能开发，并通过测试
- 阶段三：书写该 API 的中英文档

# 八、影响面
为独立新增op，对其他模块没有影响

# 名词解释
无

# 附件及参考资料

无

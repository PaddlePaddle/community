# paddle.sparse.concat 设计文档


| API名称      | paddle.sparse.concat                     |
|-------------|-----------------------------------------|
| 提交作者     | Lijingkai2023                             |
| 提交时间     | 2023-04-06                              |
| 版本号       | V1.0.0                                  |
| 依赖飞桨版本  | develop                                 |
| 文件名       | 20230406_api_design_for_sparse_concat.md |

# 一、概述
## 1、相关背景

简单来说，稀疏 Tensor 是元素大部分为零的矩阵，在实际求解任务时经常出现大规模的稀疏 Tensor。由于其自身的稀疏性，为了节省存储空间，通常会修改稀疏 Tensor 的存储结构。目前比较普遍的存储结构为 COO 和 CSR。

Paddle 目前已经实现了 COO 和 CSR 格式的稀疏 Tensor 的构建以及一些算子操作，然而目前还没有支持对其的 concat 操作，而 concat 操作在实际中是有应用价值的，因此在 Paddle 中集成该功能是有必要的。

## 2、功能目标

为 Paddle 新增 paddle.sparse.concat 稀疏 API。针对 Paddle 的两种稀疏 Tensor 格式 COO 和 CSR，都需新增 concat 的计算逻辑。一共需要新增 2 个 kernel 的前向与反向。动静态图都需要支持。

其中 COO 的 kernel 需要支持任意维度的稀疏 Tensor，CSR 的 kernel 需要支持 2D/3D 的稀疏 Tensor。

## 3、意义

支持稀疏 Tensor 的 concat 操作，丰富基础功能，提升稀疏 Tensor 的 API 完整度。

# 二、飞桨现状

目前paddle中有API 'paddle.concat(x, axis=0, name=None)'对输入沿参数 axis 轴进行联结，返回一个新的 Tensor。缺少稀疏 Tensor 的 concat 功能实现。

# 三、业内方案调研

针对 PyTorch，TensorFlow 和 SciPy 三种框架对该功能进行了调研，具体结果如下。

## PyTorch
PyTorch 中有API 'torch.cat(inputs, dimension=0) → Tensor' 在给定维度上对输入的张量序列seq 进行连接操作。
目前还不支持对稀疏 Tensor 的 cat 功能，

## TensorFlow

TensorFlow 只支持 COO 格式的 concat 功能。详情可参考官方文档（[tf.sparse.concat](https://tensorflow.google.cn/api_docs/python/tf/sparse/concat)）。
具体核心实现代码如下所示（截取自 [tensorflow/core/kernels/sparse_concat_op.cc](https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/core/kernels/sparse_concat_op.cc#L41) 和 [tensorflow/core/util/sparse/sparse_tensor.h](https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/core/util/sparse/sparse_tensor.h#L421) 文件）：

```cpp
template <typename T>
struct SparseConcatFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const OpInputList& inds,
                  const OpInputList& vals, const OpInputList& shapes,
                  int concat_dim) {
    const int N = inds.size();
    const TensorShape input_shape(shapes[0].vec<int64_t>());
    const int input_rank = input_shape.dims();

    // The input and output sparse tensors are assumed to be ordered along
    // increasing dimension number. But in order for concat to work properly,
    // order[0] must be concat_dim. So we will reorder the inputs to the
    // concat ordering, concatenate, then reorder back to the standard order.
    // We make a deep copy of the input tensors to ensure that the in-place
    // reorder doesn't create race conditions for other ops that may be
    // concurrently reading the indices and values tensors.

    gtl::InlinedVector<int64, 8> std_order(input_rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<int64_t> concat_order;
    concat_order.reserve(input_rank);
    concat_order.push_back(concat_dim);
    for (int j = 0; j < input_rank; ++j) {
      if (j != concat_dim) {
        concat_order.push_back(j);
      }
    }

    std::vector<sparse::SparseTensor> sp_inputs;
    for (int i = 0; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      sparse::SparseTensor tensor;
      OP_REQUIRES_OK(context,
                     sparse::SparseTensor::Create(
                         tensor::DeepCopy(inds[i]), tensor::DeepCopy(vals[i]),
                         current_shape, std_order, &tensor));
      sp_inputs.push_back(std::move(tensor));
      sp_inputs[i].Reorder<T>(concat_order);
    }

    sparse::SparseTensor concat = sparse::SparseTensor::Concat<T>(sp_inputs);
    concat.Reorder<T>(std_order);

    context->set_output(0, concat.indices());
    context->set_output(1, concat.values());
  }
};

template <typename T>
inline SparseTensor SparseTensor::Concat(
    const gtl::ArraySlice<SparseTensor>& tensors) {
  DCHECK_GE(tensors.size(), size_t{1}) << "Cannot concat 0 SparseTensors";
  const int dims = tensors[0].dims_;
  DCHECK_GE(dims, 1) << "Cannot concat 0-dimensional SparseTensors";
  auto order_0 = tensors[0].order();
  const int primary_dim = order_0[0];
  ShapeArray final_order(order_0.begin(), order_0.end());
  ShapeArray final_shape(tensors[0].shape().begin(), tensors[0].shape().end());
  final_shape[primary_dim] = 0;  // We'll build this up as we go along.
  int num_entries = 0;

  bool fully_ordered = true;
  for (const SparseTensor& st : tensors) {
    DCHECK_EQ(st.dims_, dims) << "All SparseTensors must have the same rank.";
    DCHECK_EQ(DataTypeToEnum<T>::v(), st.dtype())
        << "Concat requested with the wrong data type";
    DCHECK_GE(st.order()[0], 0) << "SparseTensor must be ordered";
    DCHECK_EQ(st.order()[0], primary_dim)
        << "All SparseTensors' order[0] must match.  This is the concat dim.";
    if (st.order() != final_order) fully_ordered = false;
    const VarDimArray& st_shape = st.shape();
    for (int d = 0; d < dims - 1; ++d) {
      const int cdim = (d < primary_dim) ? d : d + 1;
      DCHECK_EQ(final_shape[cdim], st_shape[cdim])
          << "All SparseTensors' shapes must match except on the concat dim.  "
          << "Concat dim: " << primary_dim
          << ", mismatched shape at dim: " << cdim
          << ".  Expecting shape like: [" << str_util::Join(final_shape, ",")
          << "] but saw shape: [" << str_util::Join(st_shape, ",") << "]";
    }

    // Update dimension of final shape
    final_shape[primary_dim] =
        (final_shape[primary_dim] + st_shape[primary_dim]);

    num_entries += st.num_entries();  // Update number of entries
  }

  // If nonconsistent ordering among inputs, set final order to -1s.
  if (!fully_ordered) {
    final_order = UndefinedOrder(final_shape);
  }

  Tensor output_ix(DT_INT64, TensorShape({num_entries, dims}));
  Tensor output_vals(DataTypeToEnum<T>::v(), TensorShape({num_entries}));

  TTypes<int64_t>::Matrix ix_t = output_ix.matrix<int64_t>();
  typename TTypes<T>::Vec vals_t = output_vals.vec<T>();

  Eigen::DenseIndex offset = 0;
  int64_t shape_offset = 0;
  for (const SparseTensor& st : tensors) {
    const int st_num_entries = st.num_entries();

    // Fill in indices & values.
    if (st_num_entries > 0) {
      std::copy_n(&st.vals_.vec<T>()(0), st_num_entries, &vals_t(offset));

      const auto* st_ix = &st.ix_.matrix<int64_t>()(0, 0);
      auto* ix_out = &ix_t(offset, 0);
      for (std::size_t i = 0; i < st_num_entries * dims; ++i) {
        *ix_out++ = *st_ix++ + ((i % dims == primary_dim) ? shape_offset : 0);
      }
    }

    offset += st_num_entries;
    shape_offset += st.shape()[primary_dim];
  }

  return SparseTensor(output_ix, output_vals, final_shape, final_order);
}
```

## SciPy

Numpy 中有'numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")' Join a sequence of arrays along an existing axis.
SciPy 不支持对稀疏 Tensor 的 concat 操作。


# 四、对比分析

由于 PyTorch 和 SciPy 并没有支持稀疏 Tensor 的 concat 操作，故我们只对 TensorFlow 进行分析。

TensorFlow 
- 优点：实现了 COO 格式下对任意维度 concat 的操作
- 缺点：仅支持 COO 格式

因此，我们可以参考 TensorFlow 的实现逻辑，来实现 COO 格式的 concat 操作。CSR 格式的 concat 操作，需自行设计实现方式。
# 五、设计思路与实现方案

## 命名与参数设计

仿照 `DenseTensor` 中 concat kernel 的设计，在 `paddle/phi/kernels/sparse/cpu/concat_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/concat_kernel.cu` 中，前向 kernel 的设计为：
```c++
template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                    const std::vector<const SparseCooTensor*>& x,
                    const Scalar& axis_scalar,
                    SparseCooTensor* out);
```
```c++
template <typename T, typename Context>
void ConcatCsrKernel((const Context& dev_ctx,
                    const std::vector<const SparseCsrTensor*>& x,
                    const Scalar& axis_scalar,
                    SparseCsrTensor* out);
```

在 `paddle/phi/kernels/sparse/cpu/concat_grad_kernel.cc` 和 `paddle/phi/kernels/sparse/gpu/concat_grad_kernel.cu` 中，反向 kernel 的设计为：
```c++
template <typename T, typename Context>
void ConcatCooGradKernel(const Context& dev_ctx,
                        const std::vector<const SparseCooTensor*>& x,
                        const SparseCooTensor& out_grad,
                        const Scalar& axis_scalar,
                        std::vector<SparseCooTensor*> x_grad);
```
```c++
template <typename T, typename Context>
void ConcatCsrGradKernel(const Context& dev_ctx,
                        const std::vector<const SparseCsrTensor*>& x,
                        const SparseCsrTensor& out_grad,
                        const Scalar& axis_scalar,
                        std::vector<SparseCsrTensor*> x_grad);
```

在 `paddle/phi/api/yaml/sparse_ops.yaml` 中新增对应 API：
```yaml
- op : concat
  args : (Tensor[] x, Scalar(int64_t) axis)
  output : Tensor(out)
  infer_meta :
    func : ConcatInferMeta
    param : [x, axis]
  kernel :
    func : concat_coo{sparse_coo -> sparse_coo},
           concat_csr{sparse_csr -> sparse_csr}
    layout: x
  backward : concat_grad
```

在 `paddle/phi/api/yaml/sparse_backward.yaml` 中新增对应 API：
```yaml
- backward_op : concat_grad
  concat (Tensor[] x, Scalar axis) -> Tensor(out)
  args : (Tensor[] x, Tensor out_grad, Scalar axis = 0)
  output : Tensor[](x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : concat_coo_grad{sparse_coo, sparse_coo -> sparse_coo},
           concat_csr_grad{sparse_csr, sparse_csr -> sparse_csr} 
```
## 底层OP设计

对于 COO 格式的 concat 操作，可以参考 TensorFlow 的方法：
- 将连接维度axis做为第一个维度，对稀疏 Tensor 的维度进行重排
- 遍历稀疏 Tensor 列表，将稀疏 Tensor 的数据顺序复制连接到一起；将稀疏 Tensor 的下标顺序复制连接到一起，并把连接维度上的下标+=已经复制的 Tensor 在连接维度上的shape
- 对稀疏 Tensor 的维度进行重排，恢复到初始维度顺序

对于 CSR 格式的 concat 操作，实现如下：
- 将连接维度axis做为第一个维度，对稀疏 Tensor 的维度进行重排
- 将稀疏 Tensor 的行起始位置(去掉最后一个数据长度)、列下标、数据分别顺序复制连接到一起；当dim==2时，行起始位置除最后一个(最后一个是长度，等于数据数组的长度)外，加上已经复制 Tensor 的数据长度
- 对稀疏 Tensor 的维度进行重排，恢复到初始维度顺序

## API实现方案

预期 Paddle 调用 concat API 的形式为：
```python
paddle.sparse.concat(x,  axis=0, name=None)
```
- **x** (list|tuple) - 待联结的稀疏 Tensor list 或者稀疏 Tensor tuple，稀疏 Tensor支持 COO 和 CSR 格式，支持的数据类型为：bool、float16、float32、float64、int32、int64、uint8， x 中所有 Tensor 的数据类型应该一致。
- **axis** (int|Tensor，可选) - 指定对输入 x 进行运算的轴，可以是整数或者形状为[1]的 Tensor，数据类型为 int32 或者 int64。 axis 的有效范围是 [-R, R)，R 是输入 x 中 Tensor 的维度，axis 为负值时与 axis+R 等价。默认值为 0。
- **name** (str，可选) - 具体用法请参见 Name，一般无需设置，默认值为 None

我们会首先检查 **x** 与 **axis** 的合法性，再进行对应的 concat 操作。

# 六、测试和验收的考量

测试考虑的 case 以及验收标准如下：

| case | 验收标准|
|------|-------|
|x的长度，x中 Tensor 的shape和数据类型对比 | 对x长度等0，或者 Tensor 的 shape 和数据类型不一致的情况能进行报错，其他情况能返回正确结果|
|axis 对边界的处理 | 对超出边界的情况能进行报错，未超出边界的情况能返回正确结果|
|不同 shape, axis 下结果的正确性 | 能返回正确结果|

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
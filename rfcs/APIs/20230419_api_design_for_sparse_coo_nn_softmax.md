# paddle.sparse.nn.Softmax 设计文档

| API名称                                                      | paddle.sparse.nn.Softmax                                   |
| ------------------------------------------------------------ | ------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-04-19                                        |
| 版本号                                                       | V1.0                                              |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                           |
| 文件名                                                       | 20230419_api_design_for_sparse_coo_nn_softmax.md<br> |


# 一、概述

## 1、相关背景

稀疏 Tensor 是元素大部分为零的矩阵，在实际求解任务时经常出现大规模的稀疏 Tensor。由于其自身的稀疏性，为了节省存储空间，通常会修改稀疏 Tensor 的存储结构。目前比较普遍的存储结构为 COO 和 CSR。

Paddle 目前已经实现了 COO 和 CSR 格式的稀疏 Tensor 的构建以及一些算子操作，Softmax目前仅仅支持了 CSR 格式的稀疏 Tensor，　还需要对COO格式的支持。

## 2、功能目标

在飞桨中增加 paddle.sparse.nn.Softmax 对COO稀疏格式的支持。

## 3、意义

飞桨将支持 paddle.sparse.nn.Softmax 在coo 稀疏格式下的计算逻辑。

# 二、飞桨现状

目前飞桨的paddle.sparse.nn.Softmax　API 仅支持CSR　格式, 还不支持COO稀疏格式。


# 三、业内方案调研

## TensorFlow

Tensorflow中提供了softmax稀疏算子支持，　详情可参考官方文档（[tf.sparse.softmax](https://tensorflow.google.cn/api_docs/python/tf/sparse/softmax))　。

```    python
tf.sparse.softmax(
    sp_input, name=None
)
```
具体核心实现代码如下所示（截取自 [tensorflow/core/kernels/sparse_softmax_op.cc](https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/core/kernels/sparse_softmax_op.cc)
```cpp
template <typename Device, typename T>
class SparseSoftmaxOp : public OpKernel {
 public:
  explicit SparseSoftmaxOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor *indices_t, *values_t, *shape_t;
    OP_REQUIRES_OK(context, context->input("sp_indices", &indices_t));
    OP_REQUIRES_OK(context, context->input("sp_values", &values_t));
    OP_REQUIRES_OK(context, context->input("sp_shape", &shape_t));

    // Validations.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument(
                    "Input sp_indices should be a matrix but received shape: ",
                    indices_t->shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(values_t->shape()) &&
                    TensorShapeUtils::IsVector(shape_t->shape()),
                errors::InvalidArgument(
                    "Inputs sp_values and sp_shape should be vectors "
                    "but received shapes: ",
                    values_t->shape().DebugString(), " and ",
                    shape_t->shape().DebugString()));
    OP_REQUIRES(context, shape_t->NumElements() >= 2,
                errors::InvalidArgument(
                    "Input should have rank >= 2, but received shape: ",
                    shape_t->SummarizeValue(3)));
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                                shape_t->flat<int64_t>(), &shape));

    const int64_t nnz = indices_t->dim_size(0);
    const int rank = static_cast<int>(indices_t->dim_size(1));
    SparseTensor st;
    OP_REQUIRES_OK(
        context, SparseTensor::Create(tensor::DeepCopy(*indices_t),
                                      tensor::DeepCopy(*values_t), shape, &st));

    Tensor *output_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nnz}),
                                                     &output_values));
    typename TTypes<T>::Flat output_flat = output_values->flat<T>();

    Tensor tmp_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   TensorShape({}), &tmp_t));
    typename TTypes<T>::Scalar tmp_scalar = tmp_t.scalar<T>();

    gtl::InlinedVector<int64_t, 4> dims(rank);
    std::iota(dims.begin(), dims.end(), 0);
    // { 0, ..., rank-1 }.
    const ArraySlice<int64_t> kReorderDims(dims);
    // All but the last dim -- the class dimension to be max-reduced along.
    const ArraySlice<int64_t> kGroupByDims = kReorderDims.subspan(0, rank - 1);
    st.Reorder<T>(kReorderDims);
    int count = 0;

    // The SparseTensor has logical shape [..., b, c], where the
    // innermost size-"c" dimension is the class dimension to be max-reduced.
    // Therefore we group by the first (rank - 1) dimensions.
    const Device &device = context->eigen_device<Device>();
    for (const auto &g : st.group(kGroupByDims)) {
      const auto group_vals = g.values<T>();
      const int group_size = group_vals.size();

      // Shifts by max, exponentiates, then renormalizes.
      tmp_scalar.device(context->eigen_device<Device>()) = group_vals.maximum();
      const T group_max = tmp_scalar();

      Eigen::Tensor<T, 1, Eigen::RowMajor> tmp(group_size);
      tmp.device(device) = (group_vals - tmp.constant(group_max)).exp();

      tmp_scalar.device(device) = tmp.sum().inverse();
      tmp.device(device) = tmp * tmp.constant(tmp_scalar());

      // Assigns back to output[count, count + group_size).
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> output_part(
          output_flat.data() + count, group_size);
      output_part.device(device) = tmp;

      count += group_size;
    }
  }
};
```

## SciPy

SciPy 不支持对稀疏 Tensor 的 softmax 操作。

## Pytorch

Pytorch中支持了softmax API的COO格式稀疏算子，　详情可参考官方文档（[torch.sparse.softmax](https://pytorch.org/docs/stable/generated/torch.sparse.softmax.html) 。
具体核心实现代码如下所示（截取自 [pytorch/src/ATen/native/sparse/SoftMax.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/sparse/SoftMax.cpp)

```    cpp
template <typename scalar_t, bool LogSoftMax>
void cpu_sparse_coo_softmax(Tensor output, const Tensor& input, const int64_t dim) {
  auto sparse_dim = input.sparse_dim();
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();
  auto out_values = output._values();
  auto out_indices = output._indices();
  out_values.resize_as_(values);
  out_indices.resize_as_(indices);
  out_indices.copy_(indices);

  if (dim >= sparse_dim) {
    if (LogSoftMax) {
      auto new_values =
          at::cpu::_log_softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    } else {
      auto new_values = at::cpu::_softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    }
    return;
  }

  auto nnz = values.size(0);
  auto sizes = input.sizes();
  auto nvalues = get_nvalues(sizes, sparse_dim);

  /* Prepare accessors */
  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.accessor<scalar_t, 2>();

  auto out_values_2 = out_values.view({nnz, nvalues});
  auto out_values_accessor = out_values_2.accessor<scalar_t, 2>();

  /* Compute independent pools of indices */
  auto pools = get_pools(indices, sizes, dim);

  int64_t grain_size = 1;
  parallel_for(0, pools.size(), grain_size, [&](int64_t begin, int64_t end) {
      for (const auto p : c10::irange(begin, end)) {
        auto pool_indices = pools[p];

        // Skip empty pools
        if (pool_indices.empty())
          continue;

        /* Prepare scratch space */
        std::vector<scalar_t> mx_row(nvalues, -std::numeric_limits<scalar_t>::infinity());
        std::vector<scalar_t> exp_sums_row(nvalues, 0);

        /* Compute mx */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            mx_row[j] = std::max(mx_row[j], values_row[j]);
          }
        }

        /* Apply exp to (v - mx) and sum the results */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          auto out_values_row = out_values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            auto v = std::exp(values_row[j] - mx_row[j]);
            if (!LogSoftMax) {
              out_values_row[j] = v;
            }
            exp_sums_row[j] += v;
          }
        }

        for (const auto j : c10::irange(nvalues)) {
          if (LogSoftMax) {
            mx_row[j] += std::log(exp_sums_row[j]);
          } else {
            exp_sums_row[j] = 1.0 / exp_sums_row[j];
          }
        }

        /* Normalize with the sum of exponents */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          auto out_values_row = out_values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            if (LogSoftMax) {
              out_values_row[j] = values_row[j] - mx_row[j];
            } else {
              out_values_row[j] *= exp_sums_row[j];
            }
          }
        }
      }
    });
}
```

# 四、对比分析

Tensorflow基于Eigen计算，支持COO稀疏格式，不支持axis传入。
Scipy没有直接支持softmax的稀疏算子计算。
Pytorch中能支持axis传入，且支持COO格式的稀疏算子。


# 五、设计思路与实现方案

## 命名与参数设计

sparse softmax 已经支持 CSR 格式，这个稀疏张量上的方法的命名和参数不需要额外设计，只需要添加相应的COO格式支持。

在 paddle/phi/api/yaml 下新增注册该算子COO格式的前向以及反向。

```    yaml
- op : softmax
  args : (Tensor x, int axis=-1)
  output : Tensor(out)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : softmax_coo{sparse_coo -> sparse_coo},
           softmax_csr{sparse_csr -> sparse_csr}
    layout : x
  backward : softmax_grad
```


```    yaml
- backward_op : softmax_grad
  forward : softmax(Tensor x, int axis=-1) -> Tensor(out)
  args : (Tensor out, Tensor out_grad, int axis)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [out]
  kernel :
    func : softmax_coo_grad{sparse_coo, sparse_coo -> sparse_coo},
           softmax_csr_grad{sparse_csr, sparse_csr -> sparse_csr}
```

## 底层OP设计

新增一个COO格式的前向以及反向Kernel：

```    cpp
template <typename T, typename Context>
void SoftmaxCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      int axis,
                      SparseCooTensor* out);
```

```    cpp
template <typename T, typename Context>
void SoftmaxCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& out,
                          const SparseCooTensor& dout,
                          int axis,
                          SparseCooTensor* dx);
```

## API实现方案

在python/paddle/sparse/nn/functional/activation.py 文件和 python/paddle/sparse/nn/layer/activation.py 文件中的原API上没有改动。

参考pytorch的计算方式，先计算索引的pool映射，在对应维度上一次计算max以及求和，最终对指数的求和进行normalize计算。

在cuda的kernel中，　若指定的axis大于等于稀疏维度，将使用稠密张量的softmax算子，若小于则分两步；
-   先计算pool和max, 基于Thrust库设计函数ComputePoolMax,　计算出指定维度上索引的pools以及每个pool对应的最大值，
-   基于pool的数量，设计对应的block和grid，调用SparseCooSoftmaxKernel, 计算pool内的softmax值
    
在反向梯度SparseCooSoftmaxGradKernel计算中，需先设计函数GetOffsets, 基于稀疏张量的索引计算对应稠密张量的偏移量，进而通过反向求导的公式计算梯度。


# 六、测试和验收的考量

完善单测代码，python/paddle/fluid/tests/unittests/test_sparse_softmax_op.py 文件中新增测试COO稀疏格式的case如下：

- 数值正确性
- COO数据格式
- 不同输入tensor的数据类型下检查输出结果
- 计算结果与dense tensor进行比较

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料

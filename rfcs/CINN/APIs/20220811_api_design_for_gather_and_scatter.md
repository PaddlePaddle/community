# CINN gather 和 scatter 设计文档

| API名称                                                      | gather/gather_nd/scatter/scatter_nd                                          |
| ---------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                             |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-08-16                                       |
| 版本号                                                        | V1.0                                             |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                        | 20220811_api_design_for_gather_and_scatter.md<br> |

# 一、概述

## 1、相关背景

`gather`和`scatter` 是众多神经网络编译器中均实现的常用算子，
`gather_nd`和`scatter_nd`是`gather`和`scatter`的多维扩展，`gather`和`scatter`互为逆运算。
假设张量 $X$尺寸为 $(16, 16, 3)$，张量 $I$尺寸为 $(16, 16, 3)$中某一维度变为12，每个元素的值均在区间 $[0, 15]$，
输入算子`gather`可以得到张量 $Y$，在其 $(i_0',i_1',i_2')$位置的值等于 $X$在 $(i_0,i_1,i_2)$位置的值，
其中 $i_{axis}=I\[(i_0',i_1',i_2')\]$, 当j不等于axis时，$i_{j}=i_j'$，`axis`参数默认值为 $0$，
此时张量 $I$尺寸为 $(12, 16, 3)$，返回的张量 $Y$与张量 $I$尺寸相同，
`gather_nd`可以指定多个`axis`，相应的 $i$也要增加1个大小为`axis`个数的维度，若未指定`axis`，
会根据 $i$的尺寸自定推算`axis`，选取前n维。
scatter与gather类似，互为逆运算具体公式见功能目标部分。
为了提升 CINN API 丰富度，需要扩充 API `gather`和`scatter`。

## 2、名词解释

- 张量/Tensor：指高维数组。
- axis/dim：指张量的维度。
- axes/dims：若干维度。
- index：索引张量。
- src：源张量，在scatter中表示取值的张量，相当于gather的计算结果。

## 3、功能目标

实现 scatter/gather 功能。

### 1.1) gather的公式表达如下

给定index, input, d<br/>
output_indices = $(i_0,...,i_{K-1})$ <br/>
index_indices = $(i_0, ..., i_{d-1}, i_{d+1}...,i_{K-1})$ <br/>

output\[ output_indices\]=input\[ $i_0, ..., i_{d-1}$, index\[ index_indices\], $i_{d+1},...,i_{K-1}$\]

### 1.2) gather_nd的公式表达如下

给定index, input<br/>
给定dims = $\[d_0,...,d_{M-1}\]$ <br/>
dims_set = \{ $d_k|k=0, 1, ..., M-1$\} <br/>
dims_u_set = \{ $0, ..., K-1$\}-dims_set <br/>

output_indices = $(i_0,...,i_{K-1})$ <br/>
index_indices = ( $u_1, u_2, ..., k$),  $u_d=i_d, d \in$ dims_u_set<br/>

index_set = \{index\[index_indices\] | $k=0, 1, ..., M-1$\} <br/>
input_indices = $(i_0,...,s_{d_0},...s_{d_1},...s_{d_{M-1}},...,i_{K-1})$，
其中 $s_d \in$ index_set<br/>

output\[ output_indices\]=input\[input_indices\]

### 1.3) gather 可以用gather_nd表达如下

gather_nd(dims=\[d\], input=input, index=index.unsqueeze(-1))

### 2.1) scatter的公式表达如下

output\[index\[ $(i_0,...,i_{K−1})$\]\]=src\[ $(i_0,...,i_{K-1})$\]
给定index, input, d<br/>
input_indices = $(i_0,...,i_{K−1})$ <br/>
index_indices = $(i_0, ..., i_{d-1}, i_{d+1}...,i_{K-1})$ <br/>

output\[ $i_0, ..., i_{d-1}$, index\[ index_indices\], $i_{d+1},...,i_{K-1}$\]=input\[input_indices\]

### 2.2) scatter_nd的公式表达如下

给定index, input，其中此处的input表示输出张量的原始值<br/>
给定dims = $\[d_0,...,d_{M-1}\]$ <br/>
dims_set = \{ $d_k|k=0, 1, ..., M-1$\} <br/>
dims_u_set = \{ $0, ..., K-1$\}-dims_set <br/>

input_indices = $(i_0,...,i_{K-1})$ <br/>
index_indices = ( $u_1, u_2, ..., k$),  $u_d=i_d, d \in$ dims_u_set<br/>

index_set = \{index\[index_indices\]| $k=0, 1, ..., M-1$\} <br/>
output = $(i_0,...,s_{d_0},...s_{d_1},...s_{d_{M-1}},...,i_{K−1})$，
其中 $s_d \in$ index_set<br/>

input\[ output_indices\]=src\[input_indices\]

### 2.3) scatter 可以用scatter_nd表达如下

scatter_nd(dims=\[d\], src=src, input=input, index=index.unsqueeze(-1))

### 示例

```python
index = [[0, 1, 1], [3, 2, 0]]
A = range(12).reshape([4, 3])
# [[ 0.0000,  1.0000,  2.0000],
# [ 3.0000,  4.0000,  5.0000],
# [ 6.0000,  7.0000,  8.0000],
# [ 9.0000, 10.0000, 11.0000]]
B_1 = gather( A, dim=0, index=index)  # C指为公式中x值
# [[0.0000, 4.0000, 5.0000],
# [9.0000, 7.0000, 2.0000]]
B_2 = gather( A, dim=1, index=index)  # C指为公式中x值
# [[0.0000, 1.0000, 1.0000],
# [0.0000, 5.0000, 3.0000]]
C = zero(4, 3)
B_3 = scatter( C, dim=0, index=index, src= B_1)  # C指为公式中output初始值
# [[0.0000, 0.0000, 2.0000],
# [0.0000, 4.0000, 5.0000],
# [0.0000, 7.0000, 0.0000],
# [9.0000, 0.0000, 0.0000]]
```

使用python实现代码可见 `五、设计思路与实现方案-底层OP设计`部分。

## 4、意义

为神经网络编译器 CINN 增加算子 `gather`、`gather_nd`、`scatter`、`scatter_nd`。

# 二、CINN现状

对CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现 `gather`、`gather_nd`、`scatter`、`scatter_nd` API。

# 三、业内方案调研

- [TVM](https://github.com/apache/tvm/blob/b79f9501fdba5cf286f015277aeae867081b77df/python/tvm/topi/scatter.py)：scatter_nd对不同维度分别实现了不同函数。gather通过一些计算的到适当的索引值，并取值。
  
  ```python
    @hybrid.script
    def _scatter_1d(data, indices, updates):
        out = output_tensor(data.shape, data.dtype)
        for i in range(data.shape[0]):
            out[i] = data[i]
        for i in range(indices.shape[0]):
            out[indices[i] if indices[i] >= 0 else indices[i] + data.shape[0]] = updates[i]
        return out


    @hybrid.script
    def _scatter_2d(data, indices, updates, axis):
        out = output_tensor(data.shape, data.dtype)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                out[i, j] = data[i, j]
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    out[
                        indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis], j
                    ] = updates[i, j]
        else:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    out[
                        i, indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis]
                    ] = updates[i, j]

        return out

  ```

  ```cpp

  binline Tensor gather(const Tensor& data, int axis, const Tensor& indices,
                     std::string name = "T_gather", std::string tag = kInjective) {
  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  ICHECK_GE(ndim_d, 1) << "Cannot gather from a scalar.";
  ICHECK_EQ(ndim_d, ndim_i);
  if (axis < 0) {
    axis += ndim_d;
  }
  ICHECK_GE(axis, 0);
  ICHECK_LT(axis, ndim_d);
  if (indices->shape[axis].as<IntImmNode>()) {
    size_t indices_dim_i = static_cast<size_t>(GetConstInt(indices->shape[axis]));
    ICHECK_GE(indices_dim_i, 1);
  }
  ICHECK(indices->dtype.is_int() || indices->dtype.is_uint());

  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < ndim_i; ++i) {
    out_shape.push_back(indices->shape[i]);
  }

  return compute(
      out_shape,
      [&](const Array<Var>& out_index) {
        Array<PrimExpr> indices_position;
        for (size_t i = 0; i < ndim_i; ++i) {
          indices_position.push_back(out_index[i]);
        }
        Array<PrimExpr> real_indices;
        for (size_t i = 0; i < ndim_i; ++i) {
          if (i == static_cast<size_t>(axis)) {
            real_indices.push_back(indices(indices_position));
          } else {
            real_indices.push_back(indices_position[i]);
          }
        }
        return data(real_indices);
      },
      name, tag);
  }

  ```


- [XLA](https://github.com/tensorflow/tensorflow/blob/0b6b491d21d6a4eb5fbab1cca565bc1e94ca9543/tensorflow/compiler/tf2xla/kernels/gather_scatter_ops.cc)：与TVM类似。

```cpp
class GatherOp : public XlaOpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing gather dimension numbers"));
    OP_REQUIRES_OK(
        context, context->GetAttr("indices_are_sorted", &indices_are_sorted_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> slice_sizes;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntVector("slice_sizes", &slice_sizes));
    xla::XlaOp result =
        xla::Gather(ctx->Input("operand"), ctx->Input("start_indices"), dnums_,
                    slice_sizes, indices_are_sorted_);
    ctx->SetOutput(0, result);
  }

 private:
  xla::GatherDimensionNumbers dnums_;
  bool indices_are_sorted_;
};

REGISTER_XLA_OP(Name("XlaGather").CompileTimeConstantInput("slice_sizes"),
                GatherOp);

class ScatterOp : public XlaOpKernel {
 public:
  explicit ScatterOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("update_computation", &update_computation_));
    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing scatter dimension numbers"));
    OP_REQUIRES_OK(
        context, context->GetAttr("indices_are_sorted", &indices_are_sorted_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = ctx->input_type(0);

    XlaCompiler::Argument update_computation_arg;
    update_computation_arg.kind = XlaCompiler::Argument::kParameter;
    update_computation_arg.type = dtype;
    update_computation_arg.shape = TensorShape();

    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.always_return_tuple = false;
    compile_options.is_entry_computation = false;
    XlaCompiler::CompilationResult update_computation;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, *update_computation_,
                            {update_computation_arg, update_computation_arg},
                            &update_computation));

    xla::XlaOp result =
        xla::Scatter(ctx->Input("operand"), ctx->Input("scatter_indices"),
                     ctx->Input("updates"), *update_computation.computation,
                     dnums_, indices_are_sorted_);
    ctx->SetOutput(0, result);
  }

 private:
  const NameAttrList* update_computation_;
  xla::ScatterDimensionNumbers dnums_;
  bool indices_are_sorted_;
};

REGISTER_XLA_OP(Name("XlaScatter"), ScatterOp);

```

# 四、对比分析

TVM 与 XLA 实现方案类似。

# 五、设计思路与实现方案

## 命名与参数设计

- input_tensor：输入张量
- index_tensor：输入索引张量
- axis：指定维度
- axes：指定若干维度
- src：源张量，在scatter中表示取值的张量，相当于gather的计算结果。
- name：输出名称

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/scatter.h` 里声明`scatter/scatter_nd`算子。
2. 在 `cinn/hlir/op/contrib/scatter.cc` 里实现`scatter/scatter_nd`算子和 `strategy`。
3. 在 `cinn/hlir/op/contrib/gather.h` 里声明`gather/gather_nd`算子。
4. 在 `cinn/hlir/op/contrib/gather.cc` 里实现`gather/gather_nd`算子和 `strategy`。
5. 在 `cinn/runtime/cpu/host_intrinsics.cc` 里实现`cinn_host_find_value_nd`函数和声明外部函数。
5. 在 `cinn/runtime/cuda/cinn_cuda_runtime_source.cuh` 里实现`cinn_cuda_find_value_nd`函数。
5. 在 `cinn/runtime/cuda/cuda_intrinsics.cuh` 里声明`cinn_cuda_find_value_nd`外部函数。
6. 在 `cinn/runtime/cpu/host_intrinsics_test.cc` 里添加测试。
使用python初步实现如下
```python
def gather(x, index, dim=0):
    y = torch.empty(index.shape, device='mps')

    def compute(indices: tuple):
        eval_indices = list(indices)
        eval_indices[dim] = index[indices].item()
        y[indices] = x[tuple(eval_indices)]

    for indices in product(*[range(s) for s in y.shape]):
        compute(indices)
    return y


def gather_nd(x, index, dims=None):
    x_shape = x.shape
    x_len = len(x_shape)
    index_shape = index.shape
    index_len = len(index_shape)
    n_dim = index_shape[-1]
    if dims is None:
        dims = range(n_dim)
    else:
        assert len(dims) == n_dim
    assert index_len - 1 > x_len - n_dim
    out_shape = index_shape[:-1]

    y = torch.empty(out_shape, device='mps')

    def compute(indices: tuple):
        x_indices = list(indices)
        index_indices = [0 for _ in range(index_len)]

        index_indices[:-1] = indices
        for i, dim in enumerate(dims):
            index_indices[-1] = i
            x_indices[dim] = index[tuple(index_indices)].item()
        y[indices] = x[tuple(x_indices)]

    for indices in product(*[range(s) for s in y.shape]):
        compute(indices)
    return y


def scatter(y, src, index, dim=0):
    def compute(indices: tuple):
        eval_indices = list(indices)
        eval_indices[dim] = index[indices].item()
        y[tuple(eval_indices)] = src[indices]

    for indices in product(*[range(s) for s in src.shape]):
        compute(indices)
    return y

  
def scatter_nd(y, src, index, dims=None):
    x_shape = x.shape
    index_shape = index.shape
    index_len = len(index_shape)
    n_dim = index_shape[-1]
    if dims is None:
        dims = range(n_dim)
    else:
        assert len(dims) == n_dim

    def compute(indices: tuple):
        x_indices = list(indices)
        index_indices = [0 for _ in range(index_len)]

        index_indices[:-1] = indices
        for i, dim in enumerate(dims):
            index_indices[-1] = i
            x_indices[dim] = index[tuple(index_indices)].item()
        y[tuple(x_indices)] = x[indices]

    for indices in product(*[range(s) for s in src.shape]):
        compute(indices)
    return y
```

## API实现方案

例如

```python
index = [[0, 1, 1], [3, 2, 0]]
A = range(12).reshape([4, 3])
# [[ 0.0000,  1.0000,  2.0000],
# [ 3.0000,  4.0000,  5.0000],
# [ 6.0000,  7.0000,  8.0000],
# [ 9.0000, 10.0000, 11.0000]]
B_1 = gather(A, dim=0, index=index)
# [[0.0000, 4.0000, 5.0000],
# [9.0000, 7.0000, 2.0000]]
B_2 = gather( A, dim=1, index=index)
# [[0.0000, 1.0000, 1.0000],
# [0.0000, 5.0000, 3.0000]]
C = zero(4, 3)
B_3 = scatter( C, dim=0, index=index, src= B_1)
# [[0.0000, 0.0000, 2.0000],
# [0.0000, 4.0000, 5.0000],
# [0.0000, 7.0000, 0.0000],
# [9.0000, 0.0000, 0.0000]]
```

1. 在 `cinn/frontend/net_build.h` 里声明 `BaseBuilder::Scatter`、`BaseBuilder::Gather`、`BaseBuilder::ScatterNd`和`BaseBuilder::GatherNd`。
2. 在 `cinn/frontend/net_build.cc` 里实现  `BaseBuilder::Scatter`、`BaseBuilder::Gather`、`BaseBuilder::ScatterNd`和`BaseBuilder::GatherNd`。

通过使用 Builder 类的方法调用 gather, scatter（其他类似）。

```python
builder = NetBuilder("test_basic")
a = builder.create_input(Float(32), (8, 24), "A")
i = builder.create_input(Int(32), (3, 24), "index")
b = builder.gather(a, index=i, dim=0)  # shape=(3, 24)
z = builder.create_input(Float(32), (8, 24), "Z")
z = builder.scatter(z, index=i, dim=0, scr=b) # shape=(8, 24)
```

# 六、测试和验收的考量

1. 在`cinn/hlir/op/contrib/gather_test.cc`和`cinn/hlir/op/contrib/scatter_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
2. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：预计9月5日前完成，已完成部分见 [PaddlePaddle/CINN#897](https://github.com/PaddlePaddle/CINN/pull/897)

# 八、影响面

对其他模块无影响。

# 附件及参考资料
- [TVM文档](https://github.com/apache/tvm/blob/b79f9501fdba5cf286f015277aeae867081b77df/python/tvm/topi/scatter.py)
- [XLA文档](https://github.com/tensorflow/tensorflow/blob/0b6b491d21d6a4eb5fbab1cca565bc1e94ca9543/tensorflow/compiler/tf2xla/kernels/gather_scatter_ops.cc)
- [CINN文档](https://paddlepaddle.github.io/CINN/)

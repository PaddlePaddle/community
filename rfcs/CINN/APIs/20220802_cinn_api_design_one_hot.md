# CINN one_hot 设计文档

| API 名称       | one_hot                             |
| -------------- | ----------------------------------- |
| 提交作者       | Nyakku Shigure（@SigureMo）         |
| 提交时间       | 2022-08-02                          |
| 版本号         | v1.0                                |
| 依赖 CINN 版本 | develop                             |
| 文件名         | 20220802_cinn_api_design_one_hot.md |

## 一、概述

### 1、相关背景

任务源于 PaddlePaddle Hackathon 第三期任务 [No.73：为神经网络编译器 CINN 增加 one_hot 算子](https://github.com/PaddlePaddle/Paddle/issues/44069#task73)

### 2、名词解释

在本任务中，由于输入参数本身是一个索引，因此输入参数索引用 `indices` 表示，而在 Compute 中 Lambda 函数的第一个参数常常也是 `indices`，这在 CINN 与 TVM 都很常见，因此容易引起歧义。

本 RFC 中的 `indices` 均表示输入索引，也即第一个输入参数，而非 Compute 中的输出元素的索引，如果两者冲突（如传入 Compute 的函数中），需修改 Compute 中索引命名。

### 3、功能目标

该算子根据输入的索引（`indices`），返回一个 Tensor，该 Tensor 将索引的位置标注为用户指定的一个值（`on_value`），非索引的位置标注为另一个值（`off_value`）。最常见的是 `1/0` 标值（`on_value=1`、`off_value=0`）

> 以下示例输出由 tvm.relay.one_hot 运算得到，由于完整代码过于复杂，因此使用伪码描述主要计算逻辑

```python
indices = [0, 2, 2] # 输入索引，shape = [3]
on_value = 1        # 索引位置的值
off_value = 0       # 非索引位置的值
depth = 3           # one_hot 的深度（也即总类别的个数）
axis = -1           # 用于填充的轴（由原来的一个数据填充成 depth 个数据）
dtype = "float32"   # 输出的数据类型

one_hot(
    indices,
    on_value=on_value,
    off_value=off_value,
    depth=depth,
    axis=axis,
    dtype=dtype
)
# shape = [3, 3]
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
```

`depth` 用于表示总类别个数，用于表示在需要填充的轴上一共要填充多少个数据

```python
# 其余参数不变
one_hot(
    indices, # shape = [3]
    on_value=on_value,
    off_value=off_value,
    depth=5,
    axis=axis,
    dtype=dtype
)
# shape = [3, 5]
# [[1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0.]]
```

`on_value` 和 `off_value` 索引位置的值和非索引位置的值，均只支持 0 维数据（Scalar）

```python
# 其余参数不变
one_hot(
    indices,            # shape = [3]
    on_value=233,       # shape = []
    off_value=-233,     # shape = []
    depth=depth,
    axis=axis,
    dtype=dtype
)
# shape = [3, 3]
# [[ 233. -233. -233.]
#  [-233. -233.  233.]
#  [-233. -233.  233.]]
```

`axis` 表示用于填充的轴，支持 `-1` 索引，用于表示填充于最后一根轴

`axis` 的范围为 `[-1, indices.ndim]`

```python
# 其余参数不变
one_hot(
    indices,            # shape = [3]
    on_value=on_value,
    off_value=off_value,
    depth=depth,
    axis=0,
    dtype=dtype
)
# shape = [3, 3]
# [[1. 0. 0.]
#  [0. 0. 0.]
#  [0. 1. 1.]]
```

`indices` 支持任意维度 Tensor（包含 0 维数据），输出数据维度为 `<indices outer dimensions> x depth x <indices inner dimensions>`

```python
indices = 1
one_hot(
    indices,            # shape = []
    on_value=on_value,
    off_value=off_value,
    depth=depth,
    axis=axis,
    dtype=dtype
)
# shape = [3]
# [0. 1. 0.]
```

> **Note**
>
> 由于现在 CINN 不支持 0 维数据，因此在 CINN 的实现中也是无法支持 0 维数据的，但 one_hot 功能本身应当支持 0 维数据，这与 `on_value` 与 `off_value` 相似，可以在 CINN 支持 0 维数据后再作考虑

下面是一个比较复杂的例子，输入维度为 `[A, B, C, D, E, F]`，如果 `axis = 2`，则，输出维度为 `[A, B, depth, C, D, E, F]`

```python
indices = np.random.randint(10, size=[2, 3, 4, 5, 6, 7])
one_hot(
    indices,            # shape = [2, 3, 4, 5, 6, 7]
    on_value=on_value,
    off_value=off_value,
    depth=10,
    axis=2,
    dtype=dtype
)
# shape = [2, 3, 10, 4, 5, 6, 7]
# 具体输出太多，不作完整展示，shape 是直接通过 relay.one_hot 打印得到，可复现
```

### 4、意义

增加 one_hot 算子可以提高 CINN 算子丰富度，使得前端框架对接 CINN 更加方便，也可以为直接通过 CINN 来组网的开发者提供更加便捷灵活的方式

## 二、CINN 现状

CINN 中已有算子并不能方便地实现 one_hot 算子的功能，因此需要开发一个新的 one_hot 算子。

## 三、业内方案调研

### XLA

XLA（tf2xla）具体实现如下

```cpp
// https://github.com/tensorflow/tensorflow/blob/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla/kernels/one_hot_op.cc#L32-L71
// tensorflow/compiler/tf2xla/kernels/one_hot_op.cc
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape indices_shape = ctx->InputShape(0);
    const TensorShape depth_shape = ctx->InputShape(1);
    const TensorShape on_value_shape = ctx->InputShape(2);
    const TensorShape off_value_shape = ctx->InputShape(3);

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth_shape),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value_shape),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value_shape),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value_shape.DebugString()));

    const int axis = (axis_ == -1) ? indices_dims : axis_;

    // The one-hot dimension.
    int64_t depth;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &depth));
    OP_REQUIRES(
        ctx, depth >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth));

    xla::XlaOp one_hot;
    OP_REQUIRES_OK(
        ctx, XlaHelpers::OneHot(ctx->builder(), depth, axis, input_type(0),
                                indices_shape, ctx->Input(0), ctx->Input(2),
                                ctx->Input(3), &one_hot));
    ctx->SetOutput(0, one_hot);
  }

// https://github.com/tensorflow/tensorflow/blob/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla/xla_helpers.cc#L90-L112
// tensorflow/compiler/tf2xla/xla_helpers.cc
Status XlaHelpers::OneHot(xla::XlaBuilder* builder, int64_t depth, int axis,
                          DataType index_type, const TensorShape& indices_shape,
                          const xla::XlaOp& indices, const xla::XlaOp& on_value,
                          const xla::XlaOp& off_value, xla::XlaOp* one_hot) {
  // Broadcast the linspace constant across the indices along the new axis,
  // and test equality at each position.
  std::vector<int64_t> broadcast_dims(indices_shape.dims());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

  TensorShape output_shape = indices_shape;
  output_shape.InsertDim(axis, depth);
  xla::Shape iota_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(index_type, output_shape, &iota_shape));

  // Selects the user-provided off_value and on_value values.
  *one_hot = xla::Select(
      xla::Eq(indices, xla::Iota(builder, iota_shape, axis), broadcast_dims),
      xla::Broadcast(on_value, output_shape.dim_sizes()),
      xla::Broadcast(off_value, output_shape.dim_sizes()));
  return OkStatus();
}
```

这包含了以下几个步骤

- 数据检查 & 预处理
  - $axis \in [-1, indices.ndim]$ 的检查
  - `depth`、`on_value`、`off_value` 均需要为 Scalar 的检查（`shape == 0`）
  - `depth` 非负检查
  - `axis == -1` 时，将其修改为 `indices.ndim`
- 算子组合
  - 计算输出 shape（`input_shape.insert(axis, depth)`）
  - 利用已有的算子进行组合，得到新的算子 one_hot（通过 `xla::Select` 来选择 `indices` 与 `xla::Iota` 生成的顺序序列一致的位置，该位置置 `on_value`，其余位置置 `off_value`）

### TVM

TVM 具体实现如下

```cpp
// https://github.com/apache/tvm/blob/832c7674fef385887a2fb99b8530736a40dfc820/src/relay/op/tensor/transform.cc#L3737-L3781
// src/relay/op/tensor/transform.cc
// relay.one_hot
TVM_REGISTER_NODE_TYPE(OneHotAttrs);

bool OneHotRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // `types` contains: [indices, on_value, off_value, result]
  ICHECK_EQ(types.size(), 4);
  const auto* indices = types[0].as<TensorTypeNode>();
  ICHECK(indices);

  const auto param = attrs.as<OneHotAttrs>();
  ICHECK_GT(param->depth, 0);

  Array<IndexExpr> oshape;
  int ndim = indices->shape.size() + 1;
  int indices_index = 0;
  int true_axis = (param->axis == -1) ? indices->shape.size() : param->axis;
  for (int i = 0; i < ndim; i++) {
    if (i == true_axis) {
      oshape.push_back(Integer(param->depth));
    } else {
      oshape.push_back(indices->shape[indices_index++]);
    }
  }

  reporter->Assign(types[3], TensorType(oshape, param->dtype));
  return true;
}


Array<te::Tensor> OneHotCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  const auto* param = attrs.as<OneHotAttrs>();
  ICHECK(param != nullptr);
  return Array<te::Tensor>{
      topi::one_hot(inputs[0], inputs[1](), inputs[2](), param->depth, param->axis, param->dtype)};
}

Expr MakeOneHot(Expr indices, Expr on_value, Expr off_value, int depth, int axis, DataType dtype) {
  auto attrs = make_object<OneHotAttrs>();
  attrs->depth = std::move(depth);
  attrs->axis = axis;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("one_hot");
  return Call(op, {indices, on_value, off_value}, Attrs(attrs), {});
}

// https://github.com/apache/tvm/blob/832c7674fef385887a2fb99b8530736a40dfc820/include/tvm/topi/transform.h#L1814-L1849
// include/tvm/topi/transform.h
inline Tensor one_hot(const Tensor& indices, const PrimExpr on_value, const PrimExpr off_value,
                      int depth, int axis, const DataType& dtype,
                      Array<PrimExpr> oshape = Array<PrimExpr>(),
                      const std::string name = "T_one_hot", const std::string tag = kInjective) {
  int true_axis = (axis == -1) ? indices->shape.size() : axis;
  if (oshape.size() == 0) {
    int ndim = indices->shape.size() + 1;
    int indices_index = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == true_axis) {
        oshape.push_back(Integer(depth));
      } else {
        oshape.push_back(indices->shape[indices_index++]);
      }
    }
  }

  PrimExpr on_value_cast = cast(dtype, on_value);
  PrimExpr off_value_cast = cast(dtype, off_value);
  return compute(
      oshape,
      [&](const Array<Var>& iter_vars) {
        Array<Var> indices_indices;
        for (size_t i = 0; i < iter_vars.size(); i++) {
          if (static_cast<int>(i) == true_axis) {
            continue;
          }

          indices_indices.push_back(iter_vars[i]);
        }

        auto idx = iter_vars[true_axis];
        return tir::Select(indices(indices_indices) == idx, on_value_cast, off_value_cast);
      },
      name, tag);
}
```

这包含了以下几个步骤

- 数据检查 & 预处理
  - 参数量检查
  - `depth` 正数检查
  - `axis == -1` 时，将其修改为 `indices.ndim`
- 计算输出 shape
  - 计算输出 shape（`input_shape.insert(axis, depth)`）
- 利用 `compute` 描述计算
  - 计算输出数据在原来的输入数据（`indices`）下的索引（`indices_indices`）（通过循环跳过 `axis` 位置的数据），这样就可以找到该索引下对应的原始的输入数据（`indices(indices_indices)`）
  - 找到输出数据索引在该轴位置的索引（`idx`）
  - 通过 `tir::Select` 进行选择，如果输出索引对应位置的输入数据（`indices(indices_indices)`）与该轴位置索引（`idx`）一致的，则置为 `on_value`，否则置为 `off_value`

## 四、对比分析

### API 设计

XLA 设计的算子如下：

```cpp
static Status OneHot(xla::XlaBuilder* builder, int64_t depth, int axis,
                       DataType index_type, const TensorShape& indices_shape,
                       const xla::XlaOp& indices, const xla::XlaOp& on_value,
                       const xla::XlaOp& off_value, xla::XlaOp* one_hot);
```

TVM 设计的 topi API 和 relay API 分别如下，算子与 API 参数一致：

```python
tvm.topi.one_hot(
  indices: tvm.te.Tensor,
  on_value: tvm.te.Tensor,
  off_value: tvm.te.Tensor,
  depth: int,
  axis: int,
  dtype: relay.DataType
)

tvm.relay.one_hot(
  indices: relay.Expr,
  on_value: relay.Expr,
  off_value: relay.Expr,
  depth: int | relay.Expr,
  axis: int,
  dtype: str
)
```

两者 API / 算子参数基本一致

### 数据检查

- `axis`：允许的范围一致，均为 `[-1, indices.ndim]`，`-1` 时的语义及处理方式一致
- `depth`：XLA 允许为 `0`，TVM 不允许，对于为 `0` 的情况，确实没有明确的语义
- `on_value`、`off_value`：XLA 对其进行了为 Scalar 的检查，这是有必要的（TVM 在它们非 Scalar 的情况时在其他算子抛出了错误）

### 核心实现

XLA 是通过算子的组合方式实现的，TVM 是通过写 Compute 描述计算

虽然两者实现方式不一样，但是两者都主要用了 `Select` 算子来筛选某一个位置与索引（`indices`）是否一致，只不过 XLA 是整体选择，因此需要配合广播，而 TVM 是逐位置选择

## 五、设计思路与实现方案

### 命名与参数设计

由于 XLA 与 TVM 均使用 `OneHot` 作为算子名称，因此直接使用 `OneHot` 作为算子名称，没有相近名称或相近操作的算子，因此没有任何歧义

设计算子参数如下：

```cpp
// cinn/hlir/op/contrib/one_hot.h
ir::Tensor OneHot(const ir::Tensor& indices,
                  const ir::Tensor& on_value,
                  const ir::Tensor& off_value,
                  const int depth,
                  const int axis,
                  const std::string& dtype,
                  const std::string& output_name);
```

### 底层 OP 设计

参考 TVM 的计算模式，利用 `lang::Compute` 来描述每一个位置的计算

- `OneHot`

  - 数据检查 & 预处理
    - `axis` 在 `[-1, indices.ndim]`
    - `depth` 大于 `0`
    - `on_value`、`off_value` 为 Scalar
    - `axis == -1` 时，将其修改为 `indices.ndim`
  - 计算输出 shape（`input_shape.insert(axis, depth)`）
  - 利用 `lang::Compute` 描述计算（基本与 TVM 一致，主要利用 `ir::Select::Make` 进行选择）

- `InferShapeForOneHot`

  - 对数据 shape 进行检查
    - 对 shape 数量进行检查
    - `on_value`、`off_value` 为 Scalar
  - 计算输出 shape（`input_shape.insert(axis, depth)`）

- `InferDtypeForOneHot`

  - 与参数 `dtype` 一致

- `StrategyForOneHot`

  - 包装 `OneHot` 返回的 compute，添加 Stages
  - 编写 schedule 接收 compute 的输出
  - 注册 compute 和 schedule

### API 实现方案

API 设计如下：

```cpp
// cinn/frontend/net_builder.h
class NetBuilder {
  Variable OneHot(const Variable& indices,
                  const Variable& on_value,
                  const Variable& off_value,
                  const int depth,
                  const int axis            = -1
                  const std::string& dtype  = "float32");
};
```

Python 端调用示例如下：

```python
builder = frontend.NetBuilder(name="one_hot")

indices = builder.create_input(type=common.Float(32), shape=(3, ), id_hint="A")
on_value = builder.create_input(type=common.Float(32), shape=(), id_hint="B")
off_value = builder.create_input(type=common.Float(32), shape=(), id_hint="C")

res = builder.one_hot(indices, on_value, off_value, depth=3, axis=-1, dtype="float32")
```

## 六、测试和验收的考量

`cinn/hlir/op/contrib/one_hot_test.cc` 主要包含一些编译时的检测，主要检测是否能正常 codegen。

`cinn/frontend/net_builder_test.cc` 主要包含一些运行时的检测，主要检测输出结果是否正确。

两者在检测时都需要考虑覆盖以下情况：

- `axis` 的各种取值，包括 `-1` 以及正数（含上界 `indices.ndim`）
- `indices` 高维度输入的正确性
- `depth` 的各种取值
- `dtype` 的各种取值最后输出的数据类型的正确性
- CPU 和 GPU 设备

## 七、可行性分析和排期规划

该算子所需的 API 在 CINN 中均已存在，但 CINN 目前不支持 0 维数据，而 `on_value` 和 `off_value` 应当是 Scalar，因此暂时使用仅含一个元素的一维数据来代替，相关检查也会对此进行检查，并在注释中说明情况，在将来 CINN 支持 0 维数据后再修改

具体规划为：

- 基本算子编写和注册「半周内，已完成」
  - 编写 Compute、InferShape、InferDtype、Strategy，并注册算子
  - 向前端 NetBuilder 添加该 API
  - 通过 pybind 暴露该 API 到 Python 端
- 调试算子，处理边界情况，编写单测「一周内」
- 向前端 Paddle 添加该算子「一周内」

## 八、影响面

对现有的其余模块无影响。

## 附件及参考资料

1. [TVM source](https://github.com/apache/tvm)
2. [tf2xla source](https://github.com/tensorflow/tensorflow/tree/7f26c09cb6b529e6f61e5d202a3419eae303364f/tensorflow/compiler/tf2xla)
3. [CINN docs](https://paddlepaddle.github.io/CINN/)
4. [CINN IR AST Doc](https://github.com/PaddlePaddle/CINN/pull/775/files)

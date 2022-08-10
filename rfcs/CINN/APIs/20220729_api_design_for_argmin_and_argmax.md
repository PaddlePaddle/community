# CINN argmax 和 argmin 设计文档

| API名称                                                      | argmax/argmin                                          |
| ---------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                             |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-07-11                                       |
| 版本号                                                        | V1.0                                             |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                        | 20220729_api_design_for_argmin_and_argmax.md<br> |

# 一、概述

## 1、相关背景

`argmax`和`argmin` 是众多神经网络编译器中基础的算子。
假设输入为 $x$，尺寸为 $(256, 256, 3)$，输入算子`argmax/argmin`可以得到张量 $x$取得最大值/最小值时的索引值，当未指定`axis`参数时，返回索引为将张量拉平时的索引数值，当指定`axis`参数时，只在指定维度上进行比较，返回最大值的索引，例如当`axis=1`时，返回的张量尺寸为 $(256, 3)$。
为了提升 CINN API 丰富度，需要扩充 API `argmax`和`argmin`。

## 2、名词解释

- 张量/Tensor：指高维数组。
- argmax/argmin：指数组或张量取得最大值/最小值时的索引值。
- axis：指张量的维度。

## 3、功能目标

实现 argmax 功能，删除张量指定尺寸为一的维度。例如，对于张量 $A$ = range(9).reshape([3, 3])，
argmax( $A$, axis = None) 结果为 $8$，argmax( $A$, axis = 1) 结果为 [2, 2, 2]，argmax( A, axis = 1，keepdim=True) 结果为[[2, 2, 2]]。

## 4、意义

为神经网络编译器 CINN 增加基础算子`argmax`和`argmin`。

# 二、CINN现状

对CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现 `argmax`和`argmin`API。

# 三、业内方案调研

- [TVM](https://github.com/apache/tvm/blob/6df070aac6d0e26d1e127095a323c61c2287eb9d/include/tvm/topi/reduction.h)：整体上通过实现fcombine和fidentity方法，传入CommReduceIdx类。以argmax为例，fcombine输入两个索引值对，比较之间的值，返回更大的索引值对。
  
  ```cpp
  inline Tensor CommReduceIdx(const Tensor& data, const Array<Integer>& axis, FCommReduce func,
                              bool keepdims, bool atleast1d) {
    auto ndim = data->shape.size();
    ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
    auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
    auto reduce_axes = MakeReduceAxes(real_axis, data);
    auto target_shape = MakeReduceTargetShape(real_axis, data, keepdims, atleast1d);
  
    auto compute = [ndim, keepdims, &real_axis, &reduce_axes, &func,
                    &data](const Array<Var>& indices) {
      Array<PrimExpr> eval_range;
      Array<PrimExpr> eval_indices;
      int arg_counter = 0;
      int red_counter = 0;
  
      for (size_t i = 0; i < ndim; ++i) {
        if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
          // real_axis contains i
          eval_range.push_back(reduce_axes[red_counter]);
          eval_indices.push_back(reduce_axes[red_counter]->var);
          red_counter++;
        } else {
          if (!keepdims) {
            eval_range.push_back(indices[arg_counter]);
            arg_counter++;
          } else {
            eval_range.push_back(indices[i]);
          }
        }
      }
  
      Array<PrimExpr> ravel_shape;
      for (auto i : real_axis) {
        ravel_shape.push_back(data->shape[i]);
      }
      auto idx = detail::RavelIndex(eval_indices, ravel_shape);
      return func({idx, data(eval_range)}, reduce_axes, nullptr);
    };
  
    auto temp_idx_val =
        tvm::te::compute(target_shape, compute, data->op->name + "_red_temp", kCommReduceIdx);
    auto temp_idx = temp_idx_val[0];
    auto temp_val = temp_idx_val[1];
    return tvm::te::compute(
        target_shape, [&temp_idx](const Array<Var>& indices) { return temp_idx(indices); },
        data->op->name + "_red", kCommReduceIdx);
  }
  inline FCommReduce MakeArgmaxReducer(bool select_last_index = false) {
    // Create a Commutative Reducer with a comparison operation, and method to get the initial value.
    auto fcombine = [=](Array<Var> lhs, Array<Var> rhs) {
      Array<PrimExpr> result;
  
      // Casting to avoid operator ambiguity
      PrimExpr lhs_idx = static_cast<PrimExpr>(lhs[0]);
      PrimExpr rhs_idx = static_cast<PrimExpr>(rhs[0]);
      PrimExpr lhs_val = static_cast<PrimExpr>(lhs[1]);
      PrimExpr rhs_val = static_cast<PrimExpr>(rhs[1]);
  
      // These variables compare the actual values of the array
      auto is_bigger = lhs_val > rhs_val;
      auto is_same = lhs_val == rhs_val;
  
      // This checks if the indices are correct for the reduction. E.g. for select_last_index
      // it gives precedence for later indices of the same element and precedence for sooner
      // indices if not select_last_index;
      PrimExpr proper_index;
      if (select_last_index) {
        proper_index = lhs_idx > rhs_idx;
      } else {
        proper_index = lhs_idx < rhs_idx;
      }
  
      PrimExpr update_index = is_bigger || (is_same && proper_index);
      result.push_back(tvm::tir::Select(update_index, lhs[0], rhs[0]));  // idx
      result.push_back(tvm::tir::Select(is_bigger, lhs[1], rhs[1]));     // val
      return result;
    };
    auto fidentity = [&](std::vector<DataType> types) {
      Array<PrimExpr> result;
      result.push_back(tvm::tir::make_const(types[0], -1));  // idx
      result.push_back(tvm::min_value(types[1]));            // val
      return result;
    };
    return MakeCommReducer(fcombine, fidentity, "argmax");
  
  }       } else {
              real_indices.push_back(0);
              flag += 1;
            }
          }
          return x(real_indices);
        },
        name, tag);
  }
  ```

- [XLA](https://github.com/pytorch/xla/blob/3d24d955b6121289a3c8bb86eda541fca7a0d69f/torch_xla/csrc/ops/arg_max.cpp)：与TVM类似。

```cpp
xla::XlaOp BuildArgMax(xla::XlaOp input, int64_t dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMax(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = torch::lazy::ToVector<int64_t>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}
```

# 四、对比分析

TVM 与 XLA 实现方案类似。

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- axis：指定维度
- keepdim：是否保持维度不变，如果未指定axis，此参数会被忽略
- name：输出名称

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/argmin.h` 里声明`argmin`算子。
2. 在 `cinn/hlir/op/contrib/argmin.cc` 里实现`argmin`算子和 `strategy`。
3. 在 `cinn/hlir/op/contrib/argmax.h` 里声明`argmax`算子。
4. 在 `cinn/hlir/op/contrib/argmax.cc` 里实现`argmax`算子和 `strategy`。

## API实现方案

例如，对于张量 A = range(9).reshape([3, 3])，
argmax( A, axis = None) 结果为8，
argmax( A, axis = 1) 结果为[2, 2, 2]。

1. 在 `cinn/frontend/net_build.h` 里声明 `BaseBuilder::ArgMax`和`BaseBuilder::ArgMin`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `BaseBuilder::ArgMax`和`BaseBuilder::ArgMin`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `BaseBuilder` 添加 `argmin/argmax` 接口，并绑定到`BaseBuilder::ArgMax`和`BaseBuilder::ArgMin`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

通过使用 Builder 类的方法调用 argmax（argmin类似）。

```python
builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (8, 24, 124), "A1")
b = builder.argmax(a)  # 输出值最大的的索引，shape=()
a = builder.create_input(Float(32), (8, 24, 124), "A2")
b = builder.argmax(a，axis=0)  # shape=(24, 124)
a = builder.create_input(Float(32), (8, 24, 124), "A3")
b = builder.argmax(a，axis=1, keepdim=True)  # shape=(8, 1, 124)
```

# 六、测试和验收的考量

1. 提供基础的 demo 文件。
2. 在`cinn/hlir/op/contrib/argmax_test.cc`和`cinn/hlir/op/contrib/argmin_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
3. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：预计8月15日前完成

# 八、影响面

对其他模块无影响。

# 附件及参考资料
[TVM文档](https://github.com/apache/tvm/blob/6df070aac6d0e26d1e127095a323c61c2287eb9d/include/tvm/topi/reduction.h)
[XLA文档](https://github.com/pytorch/xla/blob/3d24d955b6121289a3c8bb86eda541fca7a0d69f/torch_xla/csrc/ops/arg_max.cpp)
[CINN文档](https://paddlepaddle.github.io/CINN/)

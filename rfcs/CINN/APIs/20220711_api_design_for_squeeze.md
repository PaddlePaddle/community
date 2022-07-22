# CINN squeeze 设计文档

| API名称                                                      | 新增API名称                                |
| ---------------------------------------------------------- | -------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-07-11                             |
| 版本号                                                        | V1.0                                   |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                |
| 文件名                                                        | 20220711_api_design_for_squeeze.md<br> |

# 一、概述

## 1、相关背景

`squeeze` 是众多神经网络编译器中基础的算子，
例如将卷积输出$(256, 1, 1)$输入线性层中时，可以直接使 `squeeze`将维度变为$(256)$，
因此为了提升 CINN API 丰富度，需要扩充 API `squeeze`。

## 2、名词解释

张量/Tensor：指高维数组。
squeeze：指删除尺寸为1的维度，可以是指定某个维度，也可以是所有维度。
axis：指张量的维度。

## 3、功能目标

实现 squeeze 功能，删除张量指定尺寸为一的维度。

例如，对于张量 $A = (N, 1, 1, M, 1, K)$，
squeeze( $A$, axis = None) 结果尺寸为$(N, M, K)$，
squeeze( $A$, axis = 1) 结果尺寸为$(N, 1, M, 1, K)$，
squeeze( $A$, axis = [1, 2]) 结果尺寸为$(N, M, 1, K)$，且数据值不变。

## 4、意义

为神经网络编译器 CINN 增加基础算子 `squeeze`。

# 二、CINN现状

对CINN框架目前不支持此功能，可以使用 reshape API 替代，但使用 reshape API 需要明确的知道数据的尺寸，对开发者的精力消耗较大，因此有必要实现 squeeze API。

# 三、业内方案调研

- TVM：通过遍历 shape，删除为1的维度并调用 reshape 相关 API 实现。
  ```cpp
  inline Tensor squeeze(const Tensor& x, Array<Integer> axis, bool atleast1d = false,
                      std::string name = "T_squeeze", std::string tag = kInjective) {
    auto ndim = x->shape.size();
    std::vector<int> axis_val;
    if (!axis.defined() || axis.size() == 0) {
      for (size_t i = 0; i < ndim; ++i) {
        if (IsConstInt(x->shape[i]) && GetConstInt(x->shape[i]) == 1) {
          axis_val.push_back(static_cast<int>(i));
        }
      }
    } else {
      for (size_t i = 0; i < axis.size(); ++i) {
        int64_t val = axis[i]->value;
        if (val < 0) {
          val += static_cast<int>(x->shape.size());
        }
        if (IsConstInt(x->shape[val])) {
          ICHECK_EQ(GetConstInt(x->shape[val]), 1) << "Dimension " << val << " must have size 1";
        }
        axis_val.push_back(val);
      }
    }

    std::unordered_set<int> axis_set(axis_val.begin(), axis_val.end());

    Array<PrimExpr> out_shape;
    for (size_t i = 0; i < ndim; ++i) {
      if (axis_set.count(static_cast<int>(i)) == 0) {
        out_shape.push_back(x->shape[i]);
      }
    }
    if (out_shape.size() == 0 && atleast1d) {
      out_shape.push_back(1);
    }

    return compute(
        out_shape,
        [&](const Array<Var>& indices) {
          Array<PrimExpr> real_indices;
          int flag = 0;
          for (size_t i = 0; i < ndim; ++i) {
            if (axis_set.count(static_cast<int>(i)) == 0) {
              real_indices.push_back(indices[i - flag]);
            } else {
              real_indices.push_back(0);
              flag += 1;
            }
          }
          return x(real_indices);
        },
        name, tag);
  }
  ```

- XLA：通过遍历 shape，删除为1的维度并调用 reshape 相关 API 实现。
  
  ```cpp
  xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  auto output_sizes =
      BuildSqueezedDimensions(input_shape.dimensions(), /*squeeze_dim=*/-1);
  return XlaHelpers::DynamicReshape(input, output_sizes);
  }
  ```

# 四、对比分析

TVM 与 XLA 实现方案类似。

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- axis：要删除的维度集合
- name：输出名称

## 底层OP设计

1. 在 `cinn/hlir/pe/transform.cc` 里实现 `squeeze` 算子。
2. 在 `cinn/hlir/op/transform.h` 里声明相应的 `strategy`。
3. 在 `cinn/hlir/op/transform.cc` 里实现相应的 `strategy`。

## API实现方案

实现目标为对于张量 $A = (N, 1, 1, M, 1, K)$，
squeeze( $A$, axis = 1) 结果尺寸为$(N, 1, M, 1, K)$，
squeeze( $A$, axis = [1, 2]) 结果尺寸为$(N, M, 1, K)$，
squeeze( $A$, axis = None) 结果尺寸为$(N, M, K)$，且数据值不变。

1. 在 `cinn/frontend/base_build.h` 里声明 `BaseBuilder::Squeeze`。
2. 在 `cinn/frontend/base_build.cc` 里实现 `BaseBuilder::Squeeze`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `BaseBuilder` 添加 `squeeze` 接口，并绑定到 `BaseBuilder::Squeeze`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

通过使用 Builder 类的方法调用 squeeze。
```python
builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (1, 24, 16, 1, 16, 16), "A1")
b = builder.squeeze(a)  # 与 a = builder.squeeze(a，axis=None) 等价。shape=(24, 16, 16, 16)
a = builder.create_input(Float(32), (1, 24, 16, 1, 16, 16), "A2")
b = builder.squeeze(a，axis=0)  # shape=(24, 16, 1, 16, 16)
a = builder.create_input(Float(32), (1, 24, 16, 1, 16, 16), "A3")
b = builder.squeeze(a，axis=3)  # shape=(1, 24, 16, 16, 16)
a = builder.create_input(Float(32), (1, 24, 16, 1, 16, 16), "A4")
b = builder.squeeze(a，axis=4)  # raise error
```

# 六、测试和验收的考量

1. 提供基础的 demo 文件。
2. 在`cinn/hlir/pe/pe_transform_test.cc`和`cinn/hlir/op/transform_test.cc`中添加对底层OP进行测试的代码。
3. 在`python/tests`文件夹中添加对Python API进行测试的代码。
4. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：底层OP设计已完成，API实现方案即将完成，测试和文档部分预计7月20日前完成。

# 八、影响面

对其他模块无影响。

# 附件及参考资料

[CINN文档](https://paddlepaddle.github.io/CINN/)

# CINN repeat 设计文档

| API名称                                                      | 新增API名称                                |
| ---------------------------------------------------------- | -------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-09-12                             |
| 版本号                                                        | V1.0                                   |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                |
| 文件名                                                        | 20220912_api_design_for_repeat.md |

# 一、概述

## 1、相关背景

`repeat` 是众多神经网络编译器中基础的算子，
例如输入尺寸为 $(5, 6, 7)$ 的张量，可以直接使 `repeat`将指定维度进行重复，
例如对第一个维度重复3次，得到的尺寸为 $(3*5, 6, 7)$
因此为了提升 CINN API 丰富度，需要扩充 API `repeat`。

## 2、名词解释

张量/Tensor：指高维数组。
repeat_times：指重复的次数。
axis：指张量的维度。

## 3、功能目标

实现 repeat 功能，把一个张量沿某些维度重复几遍。

例如，对于张量 $A = (N, M, K)$，
repeat( $A$, repeat_times = 2, axis = 0) 结果尺寸为$(2N, M, K)$，
repeat( $A$, repeat_times = 3, axis = 1) 结果尺寸为$(N, 3M, K)$，
repeat( $A$, repeat_times = 4, axis = 2) 结果尺寸为$(N, M, 4K)$。

## 4、意义

为神经网络编译器 CINN 增加基础算子 `repeat`。

# 二、CINN现状

对CINN框架目前不支持此功能，可以使用 concat API 拼接实现张量的按维度重复，
但使用 concat API 比较麻烦，操作次数多，因此有必要实现 repeat API。

# 三、业内方案调研

- TVM：通过单独对重复维度的索引进行处理实现。
  ```cpp
  inline Tensor repeat(const Tensor& x, int repeats, int axis, std::string name = "T_repeat",
                     std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  ICHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                       << ", but got repeats = " << repeats;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }
  Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  new_shape.push_back(repeats * x->shape[axis]);
  for (size_t i = axis + 1; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
      new_shape,
      [&](const Array<Var>& indices) {
        Array<PrimExpr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(indexdiv(indices[axis], repeats));
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }
        return x(idx);
      },
      name, tag);
  }

  ```

- XLA：未实现该API。

# 四、对比分析

计划采用 TVM 的实现方案。

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- repeat_times：重复的次数
- axis：要重复的维度
- name：输出名称

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/repeat.h` 里声明`repeat`算子。
2. 在 `cinn/hlir/op/contrib/repeat.cc` 里实现`repeat`算子和 `strategy`。

## API实现方案

实现目标为对于张量 $A = (N, M, K)$，
repeat( $A$, repeat_times = 2, axis = 0) 结果尺寸为 $(2N, M, K)$，
repeat( $A$, repeat_times = 3, axis = 1) 结果尺寸为 $(N, 3M, K)$，
repeat( $A$, repeat_times = 4, axis = 2) 结果尺寸为 $(N, M, 4K)$。
repeat( $A$, axis = None) 结果尺寸为$(N, M, K)$，且数据值不变。

1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::repeat`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `netBuilder::repeat`。

# 六、测试和验收的考量

1. 在`cinn/hlir/op/contrib/repeat_test.cc`中添加对底层OP进行测试的代码，
在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
2. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：预计9月20日前完成。

# 八、影响面

对其他模块无影响。

# 附件及参考资料

[CINN文档](https://paddlepaddle.github.io/CINN/)

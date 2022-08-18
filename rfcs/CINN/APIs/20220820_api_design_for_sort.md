# CINN sort 设计文档

| API名称                                                      | sort                                    |
| ---------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                             |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-08-11                                       |
| 版本号                                                        | V1.0                                             |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                        | 20220811_api_design_for_sort.md<br> |

# 一、概述

## 1、相关背景

`sort` 是众多神经网络编译器中基础的算子。
`sort` 算子不会改变张量的尺寸，假设输入为 $x$，输入算子`sort`可以得到张量 $x$ 某纬度按一定顺序重排会后的张量，
当未指定`axis`参数时，视为第 0 个维度。
为了提升 CINN API 丰富度，需要扩充 API `sort`。

## 2、名词解释

- 张量/Tensor：指高维数组。
- sort：按从大到小或从小到大的顺序排序。
- axis：指张量的维度。

## 3、功能目标

实现 sort 功能，删除张量指定尺寸为一的维度。例如，对于张量 $A$ = range(9).reshape([3, 3])，
argmax( $A$, axis = None) 结果为 $8$，argmax( $A$, axis = 1) 结果为 [2, 2, 2]，argmax( A, axis = 1，keepdim=True) 结果为[[2, 2, 2]]。

## 4、意义

为神经网络编译器 CINN 增加基础算子`sort`。

# 二、CINN现状

对CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现 `sort`API。

# 三、业内方案调研

- [TVM](https://github.com/apache/tvm/blob/6df070aac6d0e26d1e127095a323c61c2287eb9d/include/tvm/topi/reduction.h)：整体上通过实现fcombine和fidentity方法，传入CommReduceIdx类。以argmax为例，fcombine输入两个索引值对，比较之间的值，返回更大的索引值对。
  
  ```cpp

  ```

- [XLA](https://github.com/pytorch/xla/blob/3d24d955b6121289a3c8bb86eda541fca7a0d69f/torch_xla/csrc/ops/arg_max.cpp)：与TVM类似。

```cpp

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

# CINN argsort 和 sort 设计文档

| API名称                                                      | sort                                    |
| ---------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 六个骨头                                             |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2022-08-11                                       |
| 版本号                                                        | V1.0                                             |
| 依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                        | 20220811_api_design_for_argsort_and_sort.md |

# 一、概述

## 1、相关背景

`sort` 是众多神经网络编译器中基础的算子。
`sort` 算子不会改变张量的尺寸，假设输入为 $x$，输入算子`sort`可以得到张量 $x$ 某纬度按一定顺序重排后的张量，
当未指定`axis`参数时，视为第 0 个维度。
`argsort` 是算子 `sort` 排序结果的索引值。
为了提升 CINN API 丰富度，需要扩充 API `sort`。

## 2、名词解释

- 张量/Tensor：指高维数组。
- sort：按从大到小或从小到大的顺序排序。
- axis：指张量的维度。
- is_ascend：表示增序。

## 3、功能目标

实现 sort 功能，某纬度按一定顺序重排张量。例如，对于张量 $A$ = [[5,3,9], [6,2,4]]，
sort( $A$, axis = None, is_ascend=True) 结果为 [[5,2,4], [6,3,9]]，argsort( $A$, axis = 1, is_ascend=False) 结果为 [[1,2,0], [0,2,1]]。

## 4、意义

为神经网络编译器 CINN 增加基础算子`argsort`和`sort`。

# 二、CINN现状

对CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现 `argsort`和`sort`API。

# 三、业内方案调研

- [TVM](https://github.com/apache/tvm/blob/824772489e514faf06025d7c09ce4dc13dcd7d08/src/runtime/contrib/sort/sort.cc)：主要利用std::stable_sort函数实现。
  
```cpp
template <typename DataType, typename OutType>
void sort_impl(
    DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend,
    std::function<void(OutType*, size_t, const std::pair<int64_t, DataType>&)> epilogue) {
  auto data_ptr = static_cast<DataType*>(input->data);
  auto out_ptr = static_cast<OutType*>(output->data);
  std::vector<std::pair<int64_t, DataType>> sorter;

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  for (int i = 0; i < axis_mul_before; ++i) {
    for (int j = 0; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, data_ptr[full_idx]));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<DataType>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<DataType>);
      }
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        epilogue(out_ptr, base_idx + k * axis_mul_after, sorter[k]);
      }
    }
  }
}
```

- XLA：与TVM类似。

```cpp
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSort(
    int64_t a, int64_t b, int64_t c, char** values, int32_t values_count,
    int32_t* values_primitive_type_size_in_bytes, bool is_stable,
    char* run_options, int64_t* prof_counters,
    void (*less_than)(char*, char*, char**, char**, int64_t*)) {
  // 'values' and 'values_primitive_type_size_in_bytes' are managed by the JIT
  // code, so msan can't tell they are initialized.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values, values_count * sizeof(char*));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values_primitive_type_size_in_bytes,
                                      values_count * sizeof(int32_t));

  // High-level idea of the iteration/sorting logic:
  // Conceptually we have a 3-dimensional shape [a, b, c]. b corresponds to the
  // dimension to sort, c is the product of the more minor dimensions (set to 1
  // if b is the most minor dimension), and a is the product of the more major
  // dimensions (set to 1 if b is the most major dimension). There are a * c
  // many rows that we need to sort. We iterate through these, calculate a
  // 'base_offset' value which points to the first element in that row, and add
  // i * c for accessing the 'i'-th element in that row.

  int64_t sort_dimension_elements = b;
  int64_t num_iteration_elements = a * c;
  int64_t sort_dimension_offset = c;

  std::unique_ptr<int64_t[]> indices(new int64_t[sort_dimension_elements]);
  std::unique_ptr<char*[]> comparison_values(new char*[2 * values_count]);
  std::iota(indices.get(), indices.get() + sort_dimension_elements, 0);
  std::unique_ptr<std::string[]> reordered_values(
      new std::string[sort_dimension_elements]);
  for (int64_t index = 0; index < num_iteration_elements; ++index) {
    // If the sort should be stable, we have to reinitialize indices to iota to
    // guarantee that we still keep the relative order in case of ties.
    if (is_stable && index > 0) {
      std::iota(indices.get(), indices.get() + sort_dimension_elements, 0);
    }
    // 'index' can be split into two values which index into the 'c' dimension
    // and the 'a' dimension, respectively. 'index' % 'c' is the index into the
    // 'c' dimension, 'index' / 'c' is the index into the 'a' dimension. When
    // calculating the base offset, we need to multiply the index into the 'a'
    // dimension with 'b' * 'c'.
    // 'index' / 'c' * 'c' * 'b' = ('index' - 'index' % 'c') * 'b'.
    int64_t base_offset =
        index % sort_dimension_offset +
        (index - index % sort_dimension_offset) * sort_dimension_elements;
    auto compare_function = [&](int64_t a, int64_t b) -> bool {
      for (int32_t i = 0; i < values_count; ++i) {
        int64_t memory_index_lhs = (base_offset + a * sort_dimension_offset) *
                                   values_primitive_type_size_in_bytes[i];
        int64_t memory_index_rhs = (base_offset + b * sort_dimension_offset) *
                                   values_primitive_type_size_in_bytes[i];
        comparison_values[i * 2] = values[i] + memory_index_lhs;
        comparison_values[i * 2 + 1] = values[i] + memory_index_rhs;
      }
      char result = 0;  // Overwritten by less_than.
      less_than(&result, run_options, comparison_values.get(), nullptr,
                prof_counters);
      return result != 0u;
    };
    if (is_stable) {
      std::stable_sort(indices.get(), indices.get() + sort_dimension_elements,
                       compare_function);
    } else {
      std::sort(indices.get(), indices.get() + sort_dimension_elements,
                compare_function);
    }

    // Reorder the values according to the order defined by 'indices'.
    for (int32_t idx = 0; idx < values_count; ++idx) {
      for (int64_t i = 0; i < sort_dimension_elements; ++i) {
        int64_t memory_index =
            (base_offset + indices[i] * sort_dimension_offset) *
            values_primitive_type_size_in_bytes[idx];

        reordered_values[i] =
            std::string(values[idx] + memory_index,
                        values_primitive_type_size_in_bytes[idx]);
      }
      for (int64_t i = 0; i < sort_dimension_elements; ++i) {
        int64_t memory_index = (base_offset + i * sort_dimension_offset) *
                               values_primitive_type_size_in_bytes[idx];
        memcpy(values[idx] + memory_index, reordered_values[i].c_str(),
               values_primitive_type_size_in_bytes[idx]);
      }
    }
  }
}
```

# 四、对比分析

TVM 与 XLA 实现方案类似。

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- axis：指张量的维度。
- is_ascend：表示增序。
- name：输出名称.

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/sort.h` 里声明`sort`和`argsort`算子。
2. 在 `cinn/hlir/op/contrib/sort.cc` 里实现`sort`和`argsort`算子以及 `strategy`。
1. 在 `cinn/runtime/cpu/host_intrinsics.cc` 里实现所要用到的工具函数。

## API实现方案

例如，对于张量 $A$ = [[5,3,9], [6,2,4]]，
sort( $A$, axis = None, is_ascend=True) 结果为 [[5,2,4], [6,3,9]]，
argsort( $A$, axis = 1, is_ascend=False) 结果为 [[1,2,0], [0,2,1]]。

1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::ArgSort`和`NetBuilder::Sort`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::ArgSort`和`NetBuilder::Sort`。

# 六、测试和验收的考量

1. 提供基础的 demo 文件。
2. 在`cinn/hlir/op/contrib/sort_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
3. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：预计9月5日前完成

# 八、影响面

对其他模块无影响。

# 附件及参考资料

[TVM文档](https://github.com/apache/tvm/blob/824772489e514faf06025d7c09ce4dc13dcd7d08/src/runtime/contrib/sort/sort.cc)
[XLA文档](https://github.com/tensorflow/tensorflow/blob/5e1bc2d8f8a10cf28d36d593a322e7eb4ab11780/tensorflow/compiler/xla/service/cpu/runtime_key_value_sort.cc)
[CINN文档](https://paddlepaddle.github.io/CINN/)

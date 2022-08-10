# CINN clip 设计文档

| API名称                                                      | clip                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 小张1998                                                     |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-01                                                   |
| 版本号                                                       | V1.0                                                         |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                                      |
| 文件名                                                       | 提交的markdown设计文档文件名称，如：20220802_api_design_for_clip.md<br> |


# 一、概述

## 1、相关背景

clip是众多神经网络编译器中基础的算子，属于ElementWise级别，它将输入限制在范围内，给定范围min，max，在范围内的输入则原值返回，超过max则返回max，小于min则返回min

## 2、功能目标

实现clip功能

## 3、意义

为神经网络编译器 CINN 增加基础算子clip

# 二、飞桨现状

CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现clip算子


# 三、业内方案调研

TVM的实现很简单，使用了一个lambda函数遍历tensor中的元素，用max和min配合进行clip操作：

```c++
/*!
 * \brief Creates an operation that clips each element of a tensor to
 * the interval [a_min, a_max]
 *
 * \param x The input tensor
 * \param a_min The inclusive lower bound of the interval
 * \param a_max The inclusive upper bound of the interval
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the clip operation
 */
inline Tensor clip(const Tensor& x, const PrimExpr& a_min, const PrimExpr& a_max,
                   std::string name = "T_clip", std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const Array<Var>& i) {
        auto min_val = tvm::cast(x->dtype, a_min);
        auto max_val = tvm::cast(x->dtype, a_max);
        return tvm::max(tvm::min(x(i), max_val), min_val);  // NOLINT(*)
      },
      name, tag);
}
```

xla中的实现也基本相同，不过xla封装了好多层：

```c++
XlaOp XlaBuilder::Clamp(const XlaOp& min, const XlaOp& operand,
                        const XlaOp& max) {
  return TernaryOp(HloOpcode::kClamp, min, operand, max);
}

//-----底层实现↓------
template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ElementwiseT(ElementwiseT, ElementwiseT, ElementwiseT)>
        clamp_op = [](ElementwiseT low, ElementwiseT value, ElementwiseT high) {
          return std::fmin(high, std::fmax(value, low));
        };
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clamp],
        ElementwiseTernaryOp(clamp,
                             std::move(ConvertTernaryFunction(clamp_op))));
    return Status::OK();
  }
```

# 四、对比分析

在业界，tvm和xla的实现基本相同，我们的实现也会采用相似的方法

# 五、设计思路与实现方案

## 命名与参数设计

- A：输入张量
- a_min：最小值
- a_max：最大值

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/clip.h` 里声明`clip`算子。
2. 在 `cinn/hlir/op/contrib/clip.cc` 里实现`clip`算子和 `strategy`。

## API实现方案

实现目标为对于张量 A = (M, N, K)，clip( A, a_max, a_min) 结果尺寸为 A = (M, N, K) 不变，但其中的数值发生变化，任一元素的值都在[ a_min, a_max ]的区间范围内

1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::Clip`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::Clip`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `NetBuilder` 添加 `clip` 接口，并绑定到 `NetBuilder::Clip`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

通过使用 Builder 类的方法调用 clip。

```python
builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (32, 16, 16), "A")
b = builder.clip(a，1, 100)
```

# 六、测试和验收的考量

1. 提供基础的 demo 文件。

2. 在`cinn/hlir/op/contrib/clip_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
3. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：底层OP设计已完成，API、测试和文档部分预计两周内完成

# 八、影响面

对其他模块无影响。
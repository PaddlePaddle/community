# nan_to_num 设计文档

| API名称                                                      | 新增API名称                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | paddle.nan_to_num                         |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-31                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                   |
| 文件名                                                       | 20220331_api_design_for_nan_to_num.md<br> |


# 一、概述

## 1、相关背景

nan_to_num 可将 Tensor 中 nan、正无穷、负无穷的元素替换成指定的值，避免 nan 随着计算传染到下游。

## 2、功能目标

在飞桨中增加 paddle.nan_to_num API。

## 3、意义

飞桨将支持 paddle.nan_to_num API。

# 二、飞桨现状

飞桨中还没有 nan_to_num，但可以通过类似 x[isnan(x)] = a 的方式组合。这种在 Python 层组合的方式对用户较为麻烦、易出错，且性能不高。


# 三、业内方案调研

PyTorch：PyTorch 支持 nan_to_num，CPU Kernel 实现如下：

```c++
static void nan_to_num_kernel(
    TensorIterator& iter,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "nan_to_num", [&]() {
    scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
    scalar_t pos_inf_replacement = pos_inf.has_value()
        ? static_cast<scalar_t>(pos_inf.value())
        : std::numeric_limits<scalar_t>::max();
    scalar_t neg_inf_replacement = neg_inf.has_value()
        ? static_cast<scalar_t>(neg_inf.value())
        : std::numeric_limits<scalar_t>::lowest();

    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
      return (
          at::_isnan(a)
              ? nan_replacement
              : (a == std::numeric_limits<scalar_t>::infinity()
                     ? pos_inf_replacement
                     : (a == -std::numeric_limits<scalar_t>::infinity()
                            ? neg_inf_replacement
                            : a)));
    });
  });
}
```

可以看到就是做了 naive 的匹配和替换。CUDA Kernel 和 CPU Kernel 是相似的。

NumPy：NumPy 支持 nan_to_num，实现如下：

```python
    x = _nx.array(x, subok=True, copy=copy)
    xtype = x.dtype.type

    isscalar = (x.ndim == 0)

    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x

    iscomplex = issubclass(xtype, _nx.complexfloating)

    dest = (x.real, x.imag) if iscomplex else (x,)
    maxf, minf = _getmaxmin(x.real.dtype)
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        _nx.copyto(d, nan, where=idx_nan)
        _nx.copyto(d, maxf, where=idx_posinf)
        _nx.copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x
```

可以看到也是匹配 + 替换的实现思路。

TensorFlow 不支持 nan_to_num。

# 四、对比分析

PyTorch 和 NumPy 的思路是一样的，Paddle 也可以这样实现。

# 五、设计思路与实现方案

## 命名与参数设计

```python
paddle.nan_to_num(x, nan=0.0, posinf=None, neginf=None)
```

参数和 PyTorch 对齐。nan、posinf、neginf 分别表示输入张量内值为 nan、正无穷、负无穷的元素的替换值，正无穷和负无穷默认用数据类型内最大可以表示的数字来替换。

## 底层OP设计

添加 paddle/fluid/operators/math/nan_to_num.cc，实现继承 OpProtoAndCheckerMaker 的 NanToNumOpMaker，代码类似下方：

```c++
class NanToNumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of NanToNum op,");
    AddOutput("Out", "(Tensor) The output tensor of NanToNum op,");
    AddAttr<double>("nan", "...")
        .SetDefault(0.0);
    AddAttr<bool>("replace_posinf_with_max", "Whether replace +inf with max value of the data type");
    AddAttr<double>("posinf", "Only used when 'replace_posinf_with_max' is false. Replace +inf with it.");
    AddAttr<bool>("replace_neginf_with_min", "Whether replace +inf with max value of the data type");
    AddAttr<double>("neginf", "Only used when 'replace_neginf_with_max' is false. Replace -inf with it.");
    AddComment(R"DOC(
          ...
      )DOC");
  }
};
```



并添加 paddle/phi/kernels/nan_to_num_kernel.h 文件，在其中实现计算逻辑，并添加 paddle/phi/kernels/cpu/nan_to_num_kernel.cc 和 paddle/phi/kernels/gpu/nan_to_num_kernel.cu 两个文件，它们引用 nan_to_num_kernel.h，并负责实现和注册 CPU 或 CUDA Kernel。

nan_to_num 是一个 element-wise 操作。可以使用 for_range + lambda 函数来实现，无需调用第三方库。伪代码如下：

```c++
auto numel = x->numel();
platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
for_range([x](size_t idx) { if (std::isnan(x[idx])) { x[idx] = ...; } });
```

这段伪代码会在 phi::NanToNumKernel 中调用。

## API实现方案

API 无需特殊考虑，都是 boilerplate code，代码放置在 python/paddle/tensor/math.py 文件中。

# 六、测试和验收的考量

实现一版基于 NumPy 的参考实现用于测试，预期实现效果与 NumPy 保持一致：

1. 测试 API 在动态图和静态图下与 NumPy 的一致性。
2. 测试 CPU、GPU 上与 NumPy 的一致性。
3. 测试在 fp16、fp32 和 fp64 下与 NumPy 的一致性。

# 七、可行性分析和排期规划

前两周实现代码、文档和测试。

第三周进行 Code Review 和继续迭代。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料

# paddle.linalg.matrix_exp 设计文档

| API 名称     | paddle.linalg.matrix_exp              |
| ------------ | ------------------------------------- |
| 提交作者     | xlcjz                                 |
| 提交时间     | 2023-11-29                            |
| 版本号       | V1.0                                  |
| 依赖飞桨版本 | develop                               |
| 文件名       | 20231129_api_design_for_matrix_exp.md |

# 一、概述

## 1、相关背景

Paddle 需要扩充 API：paddle.linalg.matrix_exp

## 2、功能目标

实现 `matrix_exp` API，计算方阵的指数

## 3、意义

计算方阵的指数

# 二、飞桨现状

飞桨目前没有直接提供此API，需要手动实现

# 三、业内方案调研

pytorch中有`torch.linalg.matrix_exp` API

## PyTorch

### 实现解读

在PyTorch中，matrix_exp是由c++实现的，复用了很多torch的算子API，核心代码为：

代码如下：
```cpp
// matrix exponential
Tensor mexp(const Tensor& a, bool compute_highest_degree_approx = false) {
  // squash batch dimensions to one dimension for simplicity
  const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});

  if (a.scalar_type() == at::ScalarType::Float
      || a.scalar_type() == at::ScalarType::ComplexFloat) {
    constexpr std::array<float, total_n_degs> thetas_float = {
      1.192092800768788e-07, // deg 1
      5.978858893805233e-04, // deg 2
      5.116619363445086e-02, // deg 4
      5.800524627688768e-01, // deg 8
      1.461661507209034e+00, // deg 12
      3.010066362817634e+00  // deg 18
    };

    return mexp_impl<float>(a_3d, thetas_float, compute_highest_degree_approx)
      .view(a.sizes());
  }
  else { // if Double or ComplexDouble
    constexpr std::array<double, total_n_degs> thetas_double = {
      2.220446049250313e-16, // deg 1
      2.580956802971767e-08, // deg 2
      3.397168839976962e-04, // deg 4
      4.991228871115323e-02, // deg 8
      2.996158913811580e-01, // deg 12
      1.090863719290036e+00  // deg 18
    };

    return mexp_impl<double>(a_3d, thetas_double, compute_highest_degree_approx)
      .view(a.sizes());
  }
}
```

## TensorFlow
TensorFlow 中对应的 API 函数为：`tf.linalg.expm`，通过 Python API 的方式实现。


# 四、对比分析

Tensorflow采用python组合，效率不如Pytorch
paddle中也打算用c++实现

# 五、设计思路与实现方案

## 命名与参数设计

`paddle.linalg.matrix_exp(x, name=None)`。

- `x` 输入张量: float32、float64、complex64、complex128。
- `name` 默认值`None`。

## 底层 OP 设计

使用 paddle现有 API 进行设计，不涉及底层OP新增与开发。

## API 实现方案
调研能否有现成的Paddle API复用
To be added


# 六、测试和验收的考量

1. 结果正确性:与PyTorch 中matrix_exp API计算结果保持一致
1. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

# 七、可行性分析和排期规划

可行、 工期 1month

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
# paddle.cholesky_inverse 设计文档

| API 名称 | paddle.cholesky_inverse |
| - | - |
| 提交作者 | megemini(柳顺) |
| 提交时间 | 2024-05-10 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20240510_api_design_for_cholesky_inverse.md |

# 一、概述

## 1、相关背景

使用 Cholesky 因子 `u` 计算对称正定矩阵的逆矩阵，返回矩阵 `inv`，若 `u` 为下三角矩阵，则

$$
inv = (uu^{T})^{-1}
$$

若 `u` 为上三角矩阵，则

$$
inv = (u^{T}u)^{-1}
$$

目前的 `Paddle` 框架中暂无相关 API，特在此任务中实现，涉及接口为：

- `paddle.cholesky_inverse`

以提升飞桨 API 的丰富程度。

## 2、功能目标

- `paddle.cholesky_inverse` 作为独立的函数调用
- `Tenso.cholesky_inverse` 作为 Tensor 的方法使用

## 3、意义

为 `Paddle` 增加 `cholesky_inverse` 操作，丰富 `Paddle` 中张量视图的 API。

# 二、飞桨现状

目前 `Paddle` 在 Python 端缺少相关接口的实现，而在底层也没有相关算子。

`python/paddle/tensor/math.py` 文件中实现了若干对于 `Tensor` 操作的接口，如 `inverse` 逆矩阵的计算。可以使用相关算子计算 `cholesky_inverse`。

# 三、业内方案调研

## PyTorch

`PyTorch` 实现了 `cholesky_inverse`。

参考文档 [torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch-cholesky-inverse)

`PyTorch` 底层提供 `cholesky_inverse_kernel_impl` 等函数，并暴露为上层 Python 接口 `torch.cholesky_inverse`。

其中，`torch/csrc/jit/runtime/static/generated_ops.cpp` 注册算子：

``` cpp
REGISTER_OPERATOR_FUNCTOR(
    aten::cholesky_inverse,
    aten_cholesky_inverse,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto upper = p_node->Input(1).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::cholesky_inverse(self, upper);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::cholesky_inverse_out(self, upper, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });
```

`aten/src/ATen/native/BatchLinearAlgebraKernel.cpp` 为算子代码：

``` cpp
void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "cholesky_inverse: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto batch_size = batchCount(input);
  auto n = input.size(-2);
  auto lda = std::max<int64_t>(1, n);

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* info_working_ptr = &infos_data[i];
    lapackCholeskyInverse<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
    // LAPACK writes to only upper/lower part of the matrix leaving the other side unchanged
    apply_reflect_conj_tri_single<scalar_t>(input_working_ptr, n, lda, upper);
  }
#endif
}
```

借由 `aten/src/ATen/native/BatchLinearAlgebra.cpp` 的 `lapackCholeskyInverse` 进行具体计算：

``` cpp

template<> void lapackCholeskyInverse<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotri_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotri_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<float>(char uplo, int n, float *a, int lda, int *info) {
  spotri_(&uplo, &n, a, &lda, info);
}
```

上层暴露接口为：

``` python
torch.cholesky_inverse(L, upper=False, *, out=None) → Tensor
```

其中：

- L，对称正定矩阵
- upper，是否为上三角矩阵

## MindSpore

`MindSpore` 中实现了 `cholesky_inverse`。

参考文档 [mindspore.ops.cholesky_inverse](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.cholesky_inverse.html)

接口参数与 `PyTorch` 基本一致。

# 四、对比分析

`PyTorch` 和 `MindSpore` 均实现了 `cholesky_inverse` 算子，通过底层 c++ 实现相关计算，并注册暴露为上层 Python 接口。

`TensorFlow` 未实现此接口，此处不做比对。

# 五、设计思路与实现方案

由于此接口逻辑较为简单，此处直接使用 `Paddle` 已有 Python 接口进行实现，不再添加新的 c++ 算子。

## 命名与参数设计

添加 python 上层接口:

- `paddle.cholesky_inverse(x, upper=False, name=None)`

参数：

- x: (Tensor) - 输入的对称正定矩阵。数据类型支持：float32 and float64。
- upper: (bool) - 是否为上三角矩阵, 默认为 False.

返回：

Tenso。

## 底层 OP 设计

直接使用 Python API 实现，无需设计底层 OP。

## API实现方案

利用目前 `Paddle` 已有的矩阵计算以及矩阵的逆运算进行 `cholesky_inverse`，参考代码：

``` python
In [148]: def cholesky_inverse(x, upper=False):
     ...:     if upper:
     ...:         A = x.T @ x
     ...:     else:
     ...:         A = x @ x.T
     ...:     return paddle.linalg.inv(A)

```

对比 `PyTorch` 的计算结果：

``` python
In [152]: torch.cholesky_inverse(torch.tensor([[2.,.0,.0], [4.,1.,.0], [-1.,1.,2.]]))
Out[152]:
tensor([[ 5.8125, -2.6250,  0.6250],
        [-2.6250,  1.2500, -0.2500],
        [ 0.6250, -0.2500,  0.2500]])

In [153]: cholesky_inverse(paddle.to_tensor([[2.,.0,.0], [4.,1.,.0], [-1.,1.,2.]]))
Out[153]:
Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 5.81249952, -2.62500000,  0.62499994],
        [-2.62499976,  1.25000000, -0.24999999],
        [ 0.62499976, -0.24999994,  0.24999999]])
```

两者结果一致。

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  - 常规覆盖动态图和静态图的测试场景

- **硬件场景**
  - 常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试单个向量的输入方式
  - 需要测试 Tensor.cholesky_inverse 的输入方式
  - 需要测试不同数据类型：float32, float64

- **计算精度**
  - 需要保证前向计算的精度正确性，通过 numpy 实现的函数的对比结果

- **维度测试**
  - 计算 0D，1D，2D，3D 等
  - 计算方阵，与非方阵
  - 计算上三角阵，下三角阵

- **错误用例**
  - 错误的维度
  - 错误的数据类型

# 七、可行性分析及规划排期

- 接口开发约 1 个工作日
- 接口测试约 1 个工作日

计划 1 周的工作量可以完成接口的开发预测是。

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

- [torch.cholesky_inverse](https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html#torch-cholesky-inverse)
- [mindspore.ops.cholesky_inverse](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.cholesky_inverse.html)

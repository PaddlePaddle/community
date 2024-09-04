# Adam、AdamW 优化器支持 amsgrad 设计文档

| API名称      | Adam、AdamW 优化器支持 amsgrad     |
| ------------ | ---------------------------------- |
| 提交作者     | megemini(柳顺)                     |
| 提交时间     | 2024-08-29                         |
| 版本号       | V1.0                               |
| 依赖飞桨版本 | Develop                            |
| 文件名       | 20240829_api_design_for_amsgrad.md |


# 一、概述

## 1、相关背景

> 此为 [NO.12 Adam、AdamW 优化器支持 amsgrad](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no12-adamadamw-%E4%BC%98%E5%8C%96%E5%99%A8%E6%94%AF%E6%8C%81-amsgrad) 相关设计文档。

`AMSGrad` 是 Adam 优化算法的一个变体，由 Reddi et al. 在 2018 年的论文 [《On the Convergence of Adam and Beyond》](https://openreview.net/forum?id=ryQu7f-RZ) 中提出。AMSGrad 在 Adam 的基础上引入了额外的状态变量，用于记录每个参数的历史平方梯度的最大值，这有助于算法在面对变化的梯度方差时保持稳定性。

## 2、功能目标

- 在 Paddle 的 `Adam, AdamW` 接口中，增加 `amsgrad` 选项，使其支持 `AMSGrad` 算法。
- 在 PaddleScience 的 `Adam, AdamW` 接口中，增加 `amsgrad` 选项，使其支持 `AMSGrad` 算法。

> **说明** PaddleScience 的 `Adam, AdamW` 优化器是通过调用 Paddle 的相应优化器实现，而 Paddle 的 `Adam, AdamW` 优化算法通过调用后台 c++ 算子实现，`AMSGrad` 所需要的 `历史平方梯度的最大值` 也需要在 c++ 算子中实现，因此，需要通过修改 Paddle 的 `Adam, AdamW` 优化器接口，从而支持 `AMSGrad`，而无法单独在 PaddleScience 中支持 `AMSGrad`。（单独、且只在 PaddleScience 中实现 `AMSGrad` 优化器，不在本文讨论范围之内。）

## 3、意义

Paddle 以及 PaddleScience 的 `Adam, AdamW` 优化器支持 `amsgrad` 选项。

# 二、飞桨现状

Paddle 以及 PaddleScience 的 `Adam, AdamW` 优化器暂不支持 `amsgrad` 选项，即，不支持 `AMSGrad` 优化器。

# 三、业内方案调研

## PyTorch

PyTorch 的 `Adam, AdamW` 支持 `amsgrad` 选项，以 `Adam` 为例：

- [Adam](https://pytorch.org/docs/2.4/generated/torch.optim.Adam.html#adam)

## PyTorch 接口

``` python
torch.optim.Adam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
    *,
    foreach=None,
    maximize=False,
    capturable=False,
    differentiable=False,
    fused=None)
```

具体参数说明请参考上述链接。

## PyTorch 算法

PyTorch 的算法描述为：

![image](https://github.com/user-attachments/assets/c526d9b3-218f-4c93-8214-0153ca1a4501)

如算法中所描述的，`AMSGrad` 相较于 `Adam` 的唯一不同是需要一个记录 `历史平方梯度的最大值` 的变量。

PyTorch 中 `Adam` 的代码：https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam

PyTorch 是通过 `max_exp_avg_sqs` 进行记录的，这里只摘抄关键代码：

``` python
...
if amsgrad:
    ...

    max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

    denom = (
        max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
    ).add_(eps / step_size_neg)
else:
    denom = (
        exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
    ).add_(eps / step_size_neg)
...
```

# 五、设计思路与实现方案

`Adam, AdamW` 算法对于 `amsgrad` 的实现逻辑基本一致，后文主要介绍 `Adam` 算法引入 `amsgrad` 的实现方案。

Paddle 与 PaddleScience 都是通过 `python/paddle/optimizer/adam.py` 作为主要接口实现的 `Adam` 算法。因此，需要实现以下步骤：

- `Adam` 接口引入 `amsgrad` 参数，`bool` 类型，如果为 `True` 则是 `AMSGrad` 优化器，否则仍为传统 `Adam` 优化器。
- `Adam` 接口引入新的累计变量(accumulator)： `_moment2_acc_max_str = "moment2_max"`，可参考 `_moment2_acc_str` 的实现方式，用来记录历史最大值。
- 修改 `ops.yaml` (以及相关 `yaml`)， `adam_` 以及相关算子(如 `merged_adam_` 等)引入 `amsgrad` 参数。
- 修改算子实现，如 `adam_kernel.h, adam_kernel.cc, refer.h, adam_kernel.cu` 等。
- 修改相关算子实现，如 `fused_adam_kernel.cc` 等。

> **说明：** `Adam, AdamW` 算子涉及到较多关联算子，如 `fused_adam` 等，需要一并修改。

## 命名与参数设计

```python
class Adam(Optimizer):

    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        beta1: float | Tensor = 0.9,
        beta2: float | Tensor = 0.999,
        epsilon: float | Tensor = 1e-8,
        parameters: (
            Sequence[Tensor] | Sequence[_AdamParameterConfig] | None
        ) = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        lazy_mode: bool = False,
        multi_precision: bool = False,
        use_multi_tensor: bool = False,
        amsgrad: bool = False,      # 此处为增加的接口参数
        name: str | None = None,
    ) -> None:
```

## 底层OP设计

由于 `Adam, AdamW` 涉及较多关联算子，这里仅以 `Adam` 的 `CPU` 算子为例进行说明。

### python 接口

实现路径： `python/paddle/optimizer/adam.py`

需要增加 `_moment2_acc_max_str = "moment2_max"` 变量，记录历史最大值：

``` python
self._add_accumulator(self._moment2_acc_max_str, p, dtype=acc_dtype)
```

调用 c++ 接口时传入相关变量与标记：

``` python
class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float | LRScheduler = 0.001,
        beta1: float | Tensor = 0.9,
        beta2: float | Tensor = 0.999,
        epsilon: float | Tensor = 1e-8,
        parameters: (
            Sequence[Tensor] | Sequence[_AdamParameterConfig] | None
        ) = None,
        weight_decay: float | WeightDecayRegularizer | None = None,
        grad_clip: GradientClipBase | None = None,
        lazy_mode: bool = False,
        multi_precision: bool = False,
        use_multi_tensor: bool = False,
        amsgrad: bool = False,      # 标记位
        name: str | None = None,
    ) -> None:
      ...
      self._amsgrad = amsgrad      # 标记位

  ...

  def _append_optimize_op(self, block, param_and_grad):
    ...

    _ = _C_ops.adam_(       # 调用底层算子
        param_and_grad[0],
        param_and_grad[1],
        lr,
        moment1,
        moment2,
        moment2_max,        # 输入参数，最大值
        beta1_pow_acc,
        beta2_pow_acc,
        master_weight,
        found_inf,
        _beta1,
        _beta2,
        self._epsilon,
        self._lazy_mode,
        1000,
        find_master,
        False,
        self._amsgrad,      # 标记位
    )
```

### `ops.yaml` 算子

修改算子配置文件：

``` yaml
- op : adam_
  args : (Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor moment2_max, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1 = 0.9f, Scalar beta2 = 0.999f, Scalar epsilon = 1.0e-8f, bool lazy_mode = false, int64_t min_row_size_to_use_multithread = 1000, bool multi_precision = false, bool use_global_beta_pow = false, bool amsgrad = false)
  output : Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(moment2_max_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_out)
  infer_meta :
    func : AdamInferMeta
    spmd_rule : AdamInferSpmdDynamic
  kernel :
    func : adam {dense, dense, dense, dense, dense, dense, dense, dense, dense, dense -> dense, dense, dense, dense, dense, dense, dense},
           adam_dense_param_sparse_grad {dense, selected_rows, dense, dense, dense, dense, dense, dense, dense, dense -> dense, dense, dense, dense, dense, dense, dense}
    data_type : param
  optional : master_param, skip_update, master_param_out
  inplace : (param -> param_out), (moment1 -> moment1_out), (moment2 -> moment2_out), (moment2_max -> moment2_max_out), (beta1_pow -> beta1_pow_out), (beta2_pow -> beta2_pow_out), (master_param -> master_param_out)
  traits : pir::SideEffectTrait
```

这里增加输入：

- `Tensor moment2_max`
- `bool amsgrad = false`

增加输出：

- `Tensor(moment2_max_out)`

修改 `func` 映射关系，并且，输出为 `inplace` ：

- `(moment2_max -> moment2_max_out)`

### `adam_kernel` 实现

首先需要修改算子 `Infer` 相关接口： `paddle/phi/infermeta/multiary.h`

``` cpp
void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& moment2_max,       // 增加的参数
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   const Scalar& beta1,
                   const Scalar& beta2,
                   const Scalar& epsilon,
                   bool lazy_mode,
                   int64_t min_row_size_to_use_multithread,
                   bool multi_precision,
                   bool use_global_beta_pow,
                   bool amsgrad,       // 增加的参数
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* moment2_max_out,       // 增加的参数
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs);
```

这里需要增加以上三个参数。

然后修改 `kernel` 实现： `paddle/phi/kernels/adam_kernel.h`

``` cpp
template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const DenseTensor& moment2_max,       // 增加的参数
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     const paddle::optional<DenseTensor>& master_param,
                     const paddle::optional<DenseTensor>& skip_update,
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& epsilon,
                     bool lazy_mode,
                     int64_t min_row_size_to_use_multithread,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     bool amsgrad,       // 增加的参数
                     DenseTensor* param_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* moment2_max_out,       // 增加的参数
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* master_param_outs);
```

在具体实现的地方： `paddle/phi/kernels/cpu/adam_kernel.cc`，调用了相关的实现代码：

``` cpp
    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         chunk_size,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         mom2_max_ptr + offset,       // 增加的参数
         param_ptr + offset,
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         mom2_max_out_ptr + offset,       // 增加的参数
         param_out_ptr + offset,
         amsgrad);       // 增加的参数
```

涉及到 `paddle/phi/kernels/funcs/jit/kernel_base.h` 的接口描述：

``` cpp
template <typename T>
struct AdamTuple {
  static constexpr KernelType kernel_type = kAdam;
  typedef T data_type;
  typedef adam_attr_t attr_type;
  typedef void (*func_type)(T,
                            T,
                            T,
                            T,
                            int64_t,
                            const T*,
                            const T*,
                            const T*,
                            const T*,       // 增加的参数
                            const T*,
                            T*,
                            T*,
                            T*,       // 增加的参数
                            T*,
                            bool);       // 增加的参数
};
```

具体时现为： `paddle/phi/kernels/funcs/jit/refer/refer.h`

``` cpp
template <typename T>
void Adam(T beta1,
          T beta2,
          T lr,
          T eps,
          int64_t numel,
          const T* grad_ptr,
          const T* mom1_ptr,
          const T* mom2_ptr,
          const T* mom2_max_ptr,
          const T* param_ptr,
          T* mom1_out_ptr,
          T* mom2_out_ptr,
          T* mom2_max_out_ptr,
          T* param_out_ptr,
          bool amsgrad) {
  for (int i = 0; i < numel; ++i) {
    mom1_out_ptr[i] = beta1 * mom1_ptr[i] + (1 - beta1) * grad_ptr[i];
    mom2_out_ptr[i] =
        beta2 * mom2_ptr[i] + (1 - beta2) * grad_ptr[i] * grad_ptr[i];

    // 根据 amsgrad 实现不同逻辑
    T mom2;
    if (amsgrad) {
        mom2 = std::max(mom2_out_ptr[i], mom2_max_out_ptr[i]);
        mom2_max_out_ptr[i] = mom2;
    } else {
        mom2 = mom2_out_ptr[i];
    }

    param_out_ptr[i] =
        param_ptr[i] + lr * (mom1_out_ptr[i] / (sqrt(mom2) + eps));
  }
}
```

### 分布式切分推导规则

`Adam` 的分布式切分推导规则 `spmd_rule : AdamInferSpmdDynamic` ，涉及 `paddle/phi/infermeta/spmd_rules/optimizer.h, optimizer.cc` 的修改。

同步修改上述接口：

``` c++
SpmdInfo AdamInferSpmdDynamic(const DistMetaTensor& param,
                              const DistMetaTensor& grad,
                              const DistMetaTensor& learning_rate,
                              const DistMetaTensor& moment1,
                              const DistMetaTensor& moment2,
                              const DistMetaTensor& moment2_max,       // 增加的参数
                              const DistMetaTensor& beta1_pow,
                              const DistMetaTensor& beta2_pow,
                              const DistMetaTensor& master_param,
                              const DistMetaTensor& skip_update,
                              const Scalar& beta1,
                              const Scalar& beta2,
                              const Scalar& epsilon,
                              bool lazy_mode,
                              int64_t min_row_size_to_use_multithread,
                              bool multi_precision,
                              bool use_global_beta_pow,
                              bool amsgrad)        // 增加的参数

```

由于 `moment2_max` 在 `Adam` 的执行流程中，与 `moment2` 基本一致，可以认为是伴生变量，因此，需要在 `moment2` 相应执行的地方，增加 `moment2_max` 的执行值令，如：

``` c++
  ...

  TensorDistAttr moment2_dist_attr =
      CopyTensorDistAttrForOutput(moment2.dist_attr());
  TensorDistAttr moment2_max_dist_attr =
      CopyTensorDistAttrForOutput(moment2_max.dist_attr());   // 增加的执行值令

  ...

  auto momentum2_src_dims_mapping = moment2.dist_attr().dims_mapping();
  auto momentum2_max_src_dims_mapping = moment2_max.dist_attr().dims_mapping();   // 增加的执行值令

```

## API实现方案

需要实现以下内容，包括但不限于（如果实现过程中发现更多关联算子，需要一并修改）：

Paddle 代码：

- `Adam, AdamW` 优化器上层接口与 `docstring` 增加 `amsgrad` 参数
- `ops.yaml` 算子描述
- `cpu, gpu` 的 `kernel` 实现
- 单元测试

Paddle docs 文档：

- `Adam, AdamW` 的中文文档增加 `amsgrad` 参数
- `Adam, AdamW` 转 PyTorch 接口的映射关系

PaddleScience 代码：

- `Adam, AdamW` 优化器上层接口与 `docstring` 增加 `amsgrad` 参数

# 六、测试和验收的考量

修改 `Adam, AdamW` 单元测试文件，增加 `amsgrad` 参数。

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景。

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景。

- **输入参数**
  常规覆盖 `amsgrad` 为 `False, True` 的情况。

- **输出正确性**
  输出数值结果的一致性和数据类型是否正确，比对 `numpy` 的实现正确性，线下比对 `PyTorch` 的实现正确性。

- **计算精度**
  输出数值结果的精度，比对 `numpy` 的实现精度，线下比对 `PyTorch` 的实现精度。

## 实验结果

目前，线下实现了 `amsgrad` 的 `cpu` 算子，参考论文中的实验函数：

``` python
def func(t, x):
    if t % 101 == 1:
        return (1010 * x)
    else:
        return (-10 * x)
```

调用以下实验代码：

``` python
import numpy as np

import torch
import paddle

np.random.seed(2024)

AMSGRAD = True

device = 'cpu'
iterations = 5000000

data = np.array(0).astype('float64')
samples = np.random.binomial(1, 0.01, iterations)

opt_paddle = paddle.optimizer.Adam
lr = 0.1

def func(t, x):
    if t % 101 == 1:
        return (1010 * x)
    else:
        return (-10 * x)


print(f'------ paddle ------')
paddle.set_device(device)
x = paddle.to_tensor(data)
x.stop_gradient = False
optimizer = opt_paddle(parameters=[x], learning_rate=lr, amsgrad=AMSGRAD)

for i in range(iterations):
    y = func(i, x)
    optimizer.clear_grad()
    y.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(x, y)

```

结果如下：

``` python
------ paddle ------
Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       0.10000000) Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       0.)
Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       -0.36995566) Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       -283.87649596)
Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       -1.40548992) Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       13.26455076)
Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       -2.43216356) Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
       23.61998378)
...
```

对比 PyTorch 的实验结果：

``` python
------ pytorch ------
tensor(0.1000, dtype=torch.float64, requires_grad=True) tensor(-0., dtype=torch.float64, grad_fn=<MulBackward0>)
tensor(-0.3700, dtype=torch.float64, requires_grad=True) tensor(-283.8765, dtype=torch.float64, grad_fn=<MulBackward0>)
tensor(-1.4055, dtype=torch.float64, requires_grad=True) tensor(13.2646, dtype=torch.float64, grad_fn=<MulBackward0>)
tensor(-2.4322, dtype=torch.float64, requires_grad=True) tensor(23.6200, dtype=torch.float64, grad_fn=<MulBackward0>)
...
```

两者一致，实现方案可行。

> **说明：** 论文中的实现方案，需要在每一步优化完之后对 `x` 进行 `clip(-1, 1)` 的操作，但是，目前 Paddle 的 `clip` 算子在优化过程中不能直接调用 (PyTorch 可以)，因此，上述实验代码没有 `clip` 这个步骤，但是不影响比对 PyTorch 的结果。

# 七、可行性分析和排期规划

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

[1] [NO.12 Adam、AdamW 优化器支持 amsgrad](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/%E3%80%90Hackathon%207th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no12-adamadamw-%E4%BC%98%E5%8C%96%E5%99%A8%E6%94%AF%E6%8C%81-amsgrad)

[2] [《On the Convergence of Adam and Beyond》](https://openreview.net/forum?id=ryQu7f-RZ)

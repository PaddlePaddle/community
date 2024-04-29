# paddle.optimizer.RAdam 设计文档

| API名称      | paddle.optimizer.RAdam           |
| ------------ | -------------------------------- |
| 提交作者     | megemini(柳顺)                   |
| 提交时间     | 2024-04-16                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | Develop                          |
| 文件名       | 20240416_api_design_for_RAdam.md |


# 一、概述

## 1、相关背景

> 此为 [NO.13 为 Paddle 新增 RAdam / NAdam API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no13-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-radam--nadam-api) 中 `RAdam` 算法部分。

`RAdam` (Rectified Adam) 是 Adam 算法的一个变种，通过对 Adam 的动量项进行了修正提升训练初期稳定性。具体可参考论文 [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

## 2、功能目标

在飞桨中增加 `paddle.optimizer.RAdam` 优化器

## 3、意义

飞桨用户将可以使用 `paddle.optimizer.RAdam` 优化器。

# 二、飞桨现状

飞桨通过 `python/paddle/optimizer/optimizer.py` 统一管理各种优化器，目前支持 Adam, Adamax 等优化器，暂不支持 RAdam 优化器。

# 三、业内方案调研

## PyTorch

PyTorch 支持 RAdam 优化器：

- [RADAM](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html)

## PyTorch 接口

``` python
torch.optim.RAdam(
  params,
  lr=0.001,
  betas=(0.9, 0.999),
  eps=1e-08,
  weight_decay=0,
  decoupled_weight_decay=False,
  *,
  foreach=None,
  differentiable=False)
```

具体参数说明请参考上述链接。

## PyTorch 算法

PyTorch 在具体实现时，与原论文中的算法描述基本相同，以下为原文中的算法描述：

![image](https://github.com/PaddlePaddle/community/assets/26616326/6e671528-01e4-47a7-94a5-3ca6f6512076)

PyTorch 的算法描述为：

![image](https://github.com/PaddlePaddle/community/assets/26616326/78a562da-cb2f-4ddf-bfc6-2f830357b1f0)

与原论文的不同点主要为：论文中的 $\rho _t$ 的判断条件为 $\rho _t > 4$ ，而 PyTorch 中为 $\rho _t > 5$ 。

后续以 PyTorch 的具体实现作为参考。

## PyTorch 代码

PyTorch 中 `RAdam` 的代码：https://pytorch.org/docs/stable/_modules/torch/optim/radam.html#RAdam

PyTorch 对于 optimizer 的实现方案与 Paddle 存在较大不同，其算法主体逻辑可以参考代码中 `_single_tensor_radam` 算法，由于原代码逻辑顺序与上图中算法的顺序不一致，这里仅简单摘抄部分关键代码：

``` python
# 先单独计算 bias_correction
bias_correction1 = 1 - beta1 ** step
bias_correction2 = 1 - beta2 ** step

# 计算 first and second moment
exp_avg.lerp_(grad, 1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

# 更新参数
# correcting bias for the first moving moment
bias_corrected_exp_avg = exp_avg / bias_correction1

# maximum length of the approximated SMA
rho_inf = 2 / (1 - beta2) - 1
# compute the length of the approximated SMA
rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

if rho_t > 5.0:
    # Compute the variance rectification term and update parameters accordingly
    rect = math.sqrt(
        (rho_t - 4)
        * (rho_t - 2)
        * rho_inf
        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
    )
    exp_avg_sq_sqrt = exp_avg_sq.sqrt()
    exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
    adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq_sqrt
    param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
else:
    param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)
```

另外，PyTorch 的优化器有 single_tensor 和 multi_tensor 两种实现，与算法本身无关，这里不再赘述。

# 四、对比分析

对比 Paddle 与 PyTorch 对于优化器在接口参数设计上的区别：

- Adam 优化器 (参考 Paddle docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.Adam.md)

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。仅参数名不一致。                          |
| betas     | beta1、beta2       | 一阶矩估计的指数衰减率。PyTorch 为元祖形式，Paddle 为分开的两个参数。默认值分别一致。                          |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值。仅参数名不一致。                           |
| weight_decay           | weight_decay     | 表示权重衰减系数，参数默认值不一致, PyTorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 PyTorch 一致。         |
| amsgrad   | -    | 是否使用该算法的 AMSGrad 变体。Paddle 无此参数，暂无转写方式。                       |
| foreach           | -     | 是否使用优化器的 foreach 实现。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| capturable           | -     | 在 CUDA 图中捕获此实例是否安全。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| fused      | -     | 是否使用融合实现（仅限 CUDA）。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | lazy_mode            | 设为 True 时，仅更新当前具有梯度的元素。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | multi_precision            | 是否在权重更新期间使用 multi-precision。PyTorch 无此参数，Paddle 保持默认即可。       |
| -          | use_multi_tensor            | 是否使用 multi-tensor 策略一次性更新所有参数。PyTorch 无此参数，Paddle 保持默认即可。       |

- RMSprop 优化器 (参考 Paddle docs/guides/model_convert/convert_from_pytorch/api_difference/optimizer/torch.optim.RMSprop.md)

| PyTorch                             | PaddlePaddle | 备注                                                                    |
| ----------------------------------- | ------------ | ----------------------------------------------------------------------- |
| params     | parameters           | 表示指定优化器需要优化的参数，仅参数名不一致。                      |
| lr     | learning_rate       | 学习率，用于参数更新的计算。PyTorch 默认为`0.01`，Paddle 无默认值，Paddle 需保持与 PyTorch 一致。          |
| alpha     | rho       | 平滑常数。参数默认值不一致, PyTorch 默认为`0.99`，PyTorch 默认为`0.95`，Paddle 需保持与 PyTorch 一致。     |
| eps       | epsilon        | 保持数值稳定性的短浮点类型值。参数默认值不一致, PyTorch 默认为`1e-08`，PyTorch 默认为`1e-06`，Paddle 需保持与 PyTorch 一致。  |
| weight_decay           | weight_decay     | 表示权重衰减系数。参数默认值不一致, PyTorch 默认为`0`， Paddle 默认为`None`，Paddle 需保持与 PyTorch 一致。         |
| momentum   | momentum   | 动量因子。参数完全一致。                       |
| centered   | centered   | 如果为 True，则通过梯度的估计方差，对梯度进行归一化。参数完全一致。                       |
| maximize           | -     | 根据目标最大化参数，而不是最小化。Paddle 无此参数，暂无转写方式。         |
| differentiable      | -     | 是否应通过训练中的优化器步骤进行自动微分。Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。         |
| -          | grad_clip            | 梯度裁剪的策略。 PyTorch 无此参数，Paddle 保持默认即可。       |

通过上述两个优化器参数的对比可以看到：

- Paddle 的参数设计为：

  ``` python
  class Optimizer:
      def __init__(
          self,
          learning_rate,
          parameters=None,
          weight_decay=None,
          grad_clip=None,
          name=None,
      ):
  ```

  通用参数
  - parameters，网络参数
  - learning_rate，学习率
  - weight_decay，权重衰减
  - grad_clip，梯度裁剪

  特定参数
  - beta, rho, ...，特殊作用系数
  - epsilon，稳定作用系数
  - 其他

- PyTorch 的参数设计为：

  ``` python
  class Optimizer:
      def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:
  ```

  通用参数
  - param，网络参数

  特定参数
  - defaults，参数默认值，如 lr 等
  - differentiable，自动微分
  - 其他

后续参数设计需要与 Paddle 现有优化器保持一致。对比 PyTorch ，针对 `RAdam` :

``` python
torch.optim.RAdam(
  params,
  lr=0.001,
  betas=(0.9, 0.999),
  eps=1e-08,
  weight_decay=0,
  decoupled_weight_decay=False,
  *,
  foreach=None,
  differentiable=False)
```

**需要** 的参数：

- parameters，网络参数
- learning_rate，学习率
- beta1，一阶矩估计的指数衰减率
- beta2，二阶矩估计的指数衰减率
- epsilon，稳定作用系数
- weight_decay，权重衰减
- grad_clip，梯度裁剪

**不需要** 的参数：

- decoupled_weight_decay，Adam 或 AdamW 对于 weight_decay 的处理方式。Paddle 有单独的 `LRScheduler` 处理 weight_decay ，因此，此处不需要此参数。
- foreach，是否使用优化器的 foreach 实现。参考上面的表格，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。
- differentiable，是否应通过训练中的优化器步骤进行自动微分。参考上面的表格，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。

# 五、设计思路与实现方案

Paddle 的优化器统一继承自 `python/paddle/optimizer/optimizer.py` 中的 `Optimizer` 类，此类定义并实现了优化器的通用参数与方法，如学习率 `learning_rate` 与相关的 `get_lr` ，优化器的优化操作 `step`, `minimize`，等。

自定义优化器时，可以根据具体算法的不同，实现 `Optimizer` 中的若干接口，以 `RMSProp` 为例，仅实现以下接口即可：

- `__init__` 初始化
- `_create_accumulators` 创建优化器中需要记录或累加计算的参数
- `_append_optimize_op` 将优化器所需的算子添加到网络中，此为优化器算法逻辑主要入口
- `_update_param_group` 更新各类参数

网络在优化时，网络根据 `_append_optimize_op` 中添加的算子进行计算，目前 Paddle 主要通过 `c++` 实现相应算子，如：

``` python
_C_ops.rmsprop_(
    param_and_grad[0],
    mean_square_acc,
    param_and_grad[1],
    momentum_acc,
    self._create_param_lr(param_and_grad),
    mean_grad_acc,
    master_weight,
    self._epsilon,
    self._rho,
    self._momentum,
    self._centered,
    find_master,
)
```

在更新 `param` 以及其他参数时，如 $\beta$，可以通过实现 `_finish_update` 方法，如 `Adamax` 中的做法；或直接在 `c++` 的算子中更新，如 `RMSProp` 中的做法:

``` python
inputs = {
    "Param": param_and_grad[0],
    "Grad": param_and_grad[1],
    "Moment": momentum_acc,
    "MeanSquare": mean_square_acc,
    "MeanGrad": mean_grad_acc,
    "LearningRate": self._create_param_lr(param_and_grad),
}

outputs = {
    "ParamOut": param_and_grad[0],
    "MomentOut": momentum_acc,
    "MeanSquareOut": mean_square_acc,
    "MeanGradOut": mean_grad_acc,
}

if find_master:
    inputs["MasterParam"] = master_weight
    outputs["MasterParamOut"] = master_weight
rmsprop_op = block.append_op(
    type=self.type,
    inputs=inputs,
    outputs=outputs,
    attrs={
        "epsilon": self._epsilon,
        "decay": self._rho,
        "momentum": self._momentum,
        "centered": self._centered,
    },
    stop_gradient=True,
)

return rmsprop_op

```

参考相应的算子配置文件:

``` yaml
- op : rmsprop_
  args : (Tensor param, Tensor mean_square, Tensor grad, Tensor moment, Tensor learning_rate, Tensor mean_grad, Tensor master_param, float epsilon = 1.0e-10f, float decay = 0.9f, float momentum = 0.0f, bool centered = false, bool multi_precision = false)
  output : Tensor(param_out), Tensor(moment_out), Tensor(mean_square_out), Tensor(mean_grad_out), Tensor(master_param_outs)
  infer_meta :
    func : RmspropInferMeta
  kernel :
    func : rmsprop {dense, dense, dense, dense, dense, dense, dense-> dense, dense, dense, dense, dense}
    data_type : param
  optional : mean_grad, master_param, master_param_outs
  inplace : (param -> param_out), (moment -> moment_out), (mean_square -> mean_square_out), (mean_grad -> mean_grad_out), (master_param->master_param_outs)
```

## 命名与参数设计

```python
class paddle.optimizer.RAdam(
  learning_rate=0.001,
  beta1=0.9,
  beta2=0.999,
  epsilon=1e-8,
  weight_decay=None,
  parameters=None,
  grad_clip=None,
  name=None,
)
```

和飞桨中其它优化器的风格保持一致。`weight_decay` 通过 `regularization` 参数设置，支持 L1/L2 正则。而学习率用 `LR Scheduler` 来控制，不内置在优化器内，`grad_clip` 同样为 Paddle 中 optimizer 保持一致。

## 底层OP设计

参考 `RMSProp` 等优化器，通过 Eigen 库实现 CPU Kernel，通过 for_range + Functor 实现 CUDA Kernel 。

CPU 版 Kernel 的伪代码可参考：

```c++

#include <iostream>
#include <math.h>

int main()
{
    int step = 0;
    float learning_rate = 0.01;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-08;
    float moment1 = 0;
    float moment2 = 0;
    float moment1_hat = 0;
    float rho_t = 0;
    float rho_inf = 2 / (1 - beta2) - 1;

    float g = 1e-02; // fake gradient as a constant
    float param = 0.123; // fake param, used for update

    for (int i = 1; i < 10; i++)
    {
        step += 1;
        std::cout << "step " << i << std::endl;

        moment1 = beta1 * moment1 + (1 - beta1) * g;
        moment2 = beta2 * moment2 + (1 - beta2) * g * g;

        moment1_hat = moment1 / (1 - std::pow(beta1, step));
        rho_t = rho_inf - 2 * step * std::pow(beta2, step) / (1 - std::pow(beta2, step));

        if (rho_t > 5) {
            std::cout << "rho_t > 5" << std::endl;
            float l_t = std::sqrt((1 - std::pow(beta2, step))) / (std::sqrt(moment2) + eps);
            float r_t = std::sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf)/((rho_inf - 4) * (rho_inf - 2) * rho_t));

            std::cout << "l_t " << l_t << " r_t " << r_t << std::endl;

            param = param - learning_rate * moment1_hat * r_t * l_t;
        } else {
            std::cout << "> rho_t <= 5" << std::endl;
            param = param - learning_rate * moment1_hat;
        }

        std::cout << "moment1 " << moment1 << std::endl;
        std::cout << "moment2 " << moment2 << std::endl;
        std::cout << "moment1_hat " << moment1_hat << std::endl;
        std::cout << "rho_t " << rho_t << std::endl;
        std::cout << "param " << param << std::endl;
        std::cout << "l " << param - learning_rate * moment1_hat << std::endl;
        std::cout << "====================" << std::endl;
    }
}

```

这里假设只有一个参数 `param` ，并且梯度 `grad` 为一个 `正` 的常数 `1e-05`，此时输出为：

``` shell
step 1
> rho_t <= 5
moment1 0.001
moment2 9.99987e-08
moment1_hat 0.01
rho_t 1.00001
param 0.1229
l 0.1228
====================
step 2
> rho_t <= 5
moment1 0.0019
moment2 1.99897e-07
moment1_hat 0.01
rho_t 1.99951
param 0.1228
l 0.1227
====================
step 3
> rho_t <= 5
moment1 0.00271
moment2 2.99696e-07
moment1_hat 0.01
rho_t 2.99867
param 0.1227
l 0.1226
====================
step 4
> rho_t <= 5
moment1 0.003439
moment2 3.99395e-07
moment1_hat 0.01
rho_t 3.99751
param 0.1226
l 0.1225
====================
step 5
> rho_t <= 5
moment1 0.0040951
moment2 4.98995e-07
moment1_hat 0.01
rho_t 4.99601
param 0.1225
l 0.1224
====================
step 6
rho_t > 5
l_t 99.9987 r_t 0.025821
moment1 0.00468559
moment2 5.98494e-07
moment1_hat 0.01
rho_t 5.99417
param 0.122242
l 0.122142
====================
step 7
rho_t > 5
l_t 99.9988 r_t 0.0327387
moment1 0.00521703
moment2 6.97895e-07
moment1_hat 0.01
rho_t 6.992
param 0.121914
l 0.121814
====================
step 8
rho_t > 5
l_t 99.9989 r_t 0.0387381
moment1 0.00569533
moment2 7.97195e-07
moment1_hat 0.01
rho_t 7.9895
param 0.121527
l 0.121427
====================
step 9
rho_t > 5
l_t 99.9989 r_t 0.0441046
moment1 0.0061258
moment2 8.96397e-07
moment1_hat 0.01
rho_t 8.98667
param 0.121086
l 0.120986
====================
```

可以看到，`param` 不断更新，并且，由于梯度为 `正` 且为常数，因此不断 `减小` 。

另外，可以关注到 $\rho _t$ 在 `step 5~6` 的逻辑切换，可以正常更新参数。

## API实现方案

参考其他 Paddle 中优化器的实现方案，此处需要实现：

- python 端 API `class paddle.optimizer.RAdam`
- ops.yaml 的算子配置
- c++ 端算子，CPU 与 GPU 算子
- RAdamInferMeta 推理 dtype 与 shape
- 单元测试
- 文档

# 六、测试和验收的考量

测试需考虑测试 `radam op`，以及 `RAdam API` 。

其中 `radam op` 需要继承自 `from op_test import OpTest`：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景。

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景。

- **输入参数**
  常规覆盖默认参数，常用参数，错误参数。

- **输出正确性**
  输出数值结果的一致性和数据类型是否正确，设计 `radam_step` 函数，比对 `1` 步，`多` 步的计算结果。

- **计算精度**
  需要保证 `前向/后向` 计算的精度正确性；需要比对 `_multi_precision` 的计算精度。

- **API 的正确调用**
  需要覆盖动态图和静态图的测试场景； CPU、GPU 两种测试场景；`前向/后向` 正确运行。

# 七、可行性分析和排期规划

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

[1] [NO.13 为 Paddle 新增 RAdam / NAdam API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no13-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-radam--nadam-api)

[2] [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

[3] [RADAM](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html)

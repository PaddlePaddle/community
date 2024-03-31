# paddle.optimizer.NAdam 设计文档


| API名称      | paddle.optimizer.NAdam           |
| ------------ | -------------------------------- |
| 提交作者     | megemini(柳顺)                   |
| 提交时间     | 2024-03-29                       |
| 版本号       | V1.0                             |
| 依赖飞桨版本 | Develop                          |
| 文件名       | 20240329_api_design_for_NAdam.md |


# 一、概述
## 1、相关背景
> 此为 [NO.13 为 Paddle 新增 RAdam / NAdam API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E5%BC%80%E6%BA%90%E8%B4%A1%E7%8C%AE%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no13-%E4%B8%BA-paddle-%E6%96%B0%E5%A2%9E-radam--nadam-api) 中 `NAdam` 算法部分。

`NAdam` (Nesterov-accelerated Adaptive Moment Estimation) 是 Adam 算法的一个变种，其主要改进为结合了 Nesterov 动量与 Adam 适应性学习率的优势，具体可参考论文 [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)

## 2、功能目标

在飞桨中增加 `paddle.optimizer.NAdam` 优化器

## 3、意义
飞桨用户将可以使用 `paddle.optimizer.NAdam` 优化器。

# 二、飞桨现状
飞桨通过 `python/paddle/optimizer/optimizer.py` 统一管理各种优化器，目前支持 Adam, Adamax 等优化器，暂不支持 NAdam 优化器。

# 三、业内方案调研
## PyTorch

PyTorch 支持 NAdam 优化器：

- [NADAM](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)

## PyTorch 接口

``` python
torch.optim.NAdam(
  params,
  lr=0.002,
  betas=(0.9, 0.999),
  eps=1e-08,
  weight_decay=0,
  momentum_decay=0.004,
  decoupled_weight_decay=False,
  *,
  foreach=None,
  capturable=False,
  differentiable=False)
```

具体参数说明请参考上述链接。

## PyTorch 算法

PyTorch 在具体实现时，与原论文中的算法描述略有不同，以下为原文中的算法描述：

![image](https://github.com/PaddlePaddle/Paddle/assets/26616326/e684e1b5-5664-40c8-be87-439141f3fa2b)

PyTorch 的算法描述为：

![image](https://github.com/PaddlePaddle/Paddle/assets/26616326/8d912e4b-51c2-42a8-9191-ea849cec7cb9)

与原论文的不同点主要为：

- `decdecoupled_weight_decay` 分支

  由于 Paddle 中的 `weight decay` 单独由 `python/paddle/optimizer/lr.py` 控制，因此不影响后续代码实现。

- $\mu _t$ 与 $\mu _ {t+1}$ 的具体实现方法

  在论文中，作者只是提到， $\mu$ 应该逐渐增加或减少，而没有写明具体计算方法：
  > It often helps to gradually increase or decrease µ over time, so for the rest of this section we will assume a list of values for µ indexed by timestep µ1 , . . . , µT in order to aid clarity.

  PyTorch 这里利用 `momentum decay` 对 $\beta$ 进行计算，后续 Paddle 实现中参考此处算法。

## PyTorch 代码
PyTorch 中 `NAdam` 的代码：https://pytorch.org/docs/stable/_modules/torch/optim/nadam.html#NAdam

PyTorch 对于 optimizer 的实现方案与 Paddle 存在较大不同，其算法主体逻辑可以参考代码中 `_single_tensor_nadam` 算法，由于原代码逻辑顺序与上图中算法的顺序不一致，这里仅简单摘抄部分关键代码：

``` python
# 先单独计算 bias_correction2
bias_correction2 = 1 - beta2 ** step

# 计算 momentum \mu^{t} and \mu^{t+1}
mu = beta1 * (1. - 0.5 * (0.96 ** (step * momentum_decay)))
mu_next = beta1 * (1. - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

# 计算 mu_product
mu_product *= mu

# 计算 first and second moment
exp_avg.lerp_(grad, 1 - beta1)
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
denom = exp_avg_sq.div(bias_correction2).sqrt()

# 更新参数
mu_product_next = _get_value(mu_product) * mu_next
denom.add_(eps)
param.addcdiv_(grad, denom, value=(-lr * (1. - mu) / (1. - _get_value(mu_product))))
param.addcdiv_(exp_avg, denom, value=(-lr * mu_next) / (1. - mu_product_next))
```

另外，PyTorch 的优化器有 single_tensor 和 multi_tensor 两种实现，与算法本身无关，这里不再赘述。

## TensorFlow (Keras)
TensorFlow 中的 NAdam 优化器在 Keras 中

- [tf.keras.optimizers.Nadam](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Nadam)
- [keras/optimizers/nadam.py](https://github.com/keras-team/keras/blob/v3.1.1/keras/optimizers/nadam.py#L7-L165)

## TensorFlow 接口

``` python
tf.keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    name='nadam',
    **kwargs
)
```

具体参数说明请参考上述链接。

## TensorFlow 代码
TensorFlow 的算法实现可以直接参考源码中的 `update_step` 方法：

``` python
def update_step(self, gradient, variable, learning_rate):
    """Update step given gradient and the associated model variable."""
    var_dtype = variable.dtype
    lr = ops.cast(learning_rate, var_dtype)
    gradient = ops.cast(gradient, var_dtype)

    local_step = ops.cast(self.iterations + 1, var_dtype)
    next_step = ops.cast(self.iterations + 2, var_dtype)
    decay = ops.cast(0.96, var_dtype)
    beta_1 = ops.cast(self.beta_1, var_dtype)
    beta_2 = ops.cast(self.beta_2, var_dtype)
    u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
    u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
    u_product_t = ops.cast(self._u_product, var_dtype)

    u_product_t_1 = u_product_t * u_t_1
    beta_2_power = ops.power(beta_2, local_step)

    m = self._momentums[self._get_variable_index(variable)]
    v = self._velocities[self._get_variable_index(variable)]

    self.assign_add(
        m, ops.multiply(ops.subtract(gradient, m), (1 - beta_1))
    )
    self.assign_add(
        v, ops.multiply(ops.subtract(ops.square(gradient), v), (1 - beta_2))
    )
    m_hat = ops.add(
        ops.divide(ops.multiply(u_t_1, m), 1 - u_product_t_1),
        ops.divide(ops.multiply(1 - u_t, gradient), 1 - u_product_t),
    )
    v_hat = ops.divide(v, (1 - beta_2_power))

    self.assign_sub(
        variable,
        ops.divide(
            ops.multiply(m_hat, lr), ops.add(ops.sqrt(v_hat), self.epsilon)
        ),
    )

```

其整体逻辑与 PyTorch 在算法描述中(不是代码)的逻辑顺序一致，只不过，TensorFlow 对于 $\mu _t$ 与 $\mu _ {t+1}$ 的计算方法不一样。由于原作者没有写明这两个参数如何计算，这里后续以 PyTorch 的算法为准。

# 四、对比分析
对比 PyTorch 与 TensorFlow 中 NAdam 算法，抛开 optimizer 的实现框架不同之外：

- 实现逻辑不同

  PyTorch 将部分操作整合到一起，TensorFlow 与原算法顺序一致

- $\mu _t$ 与 $\mu _ {t+1}$

  PyTorch 使用 `momentum_decay` 与一个系数  `0.96` 和 `step` 组合计算，TensorFlow 只与 `step` 相关


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
class paddle.optimizer.NAdam(
  learning_rate=0.001,
  beta1=0.9,
  beta2=0.999,
  epsilon=1e-8,
  momentum_decay=0.004,
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
    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float momentum_decay = 0.004;
    float mu_product_t = 0;
    float mu_product_t_1 = 0;
    float mu_t = 0;
    float mu_t_1 = 0;
    float m_t = 0; // first moment
    float v_t = 0; // second moment
    float m_t_hat = 0;
    float v_t_hat = 0;
    float eps = 1e-08;

    float g = 1e-05; // fake gradient as a constant
    float param = 0.123; // fake param, used for update

    for (int i = 1; i < 5; i++)
    {
        step += 1;

        mu_t = beta1 * (1.0 - 0.5 * std::pow(0.96, step * momentum_decay));
        mu_t_1 = beta1 * (1.0 - 0.5 * std::pow(0.96, (step + 1) * momentum_decay));

        mu_product_t = step == 1 ? mu_t : mu_product_t_1;
        mu_product_t_1 = mu_product_t * mu_t_1;

        m_t = beta1 * m_t + (1.0 - beta1) * g;
        v_t = beta2 * v_t + (1.0 - beta2) * g * g;

        m_t_hat = mu_t_1 * m_t / (1.0 - mu_product_t_1) + (1.0 - mu_t) * g / (1.0 - mu_product_t);
        v_t_hat = v_t / (1.0 - std::pow(beta2, step));

        param = param - learning_rate * m_t_hat / (std::sqrt(v_t_hat) + eps);

        std::cout << "step " << i << std::endl;
        std::cout << "mu_t " << mu_t << std::endl;
        std::cout << "mu_t_1 " << mu_t_1 << std::endl;
        std::cout << "mu_product_t " << mu_product_t << std::endl;
        std::cout << "mu_product_t_1 " << mu_product_t_1 << std::endl;
        std::cout << "m_t " << m_t << std::endl;
        std::cout << "v_t " << v_t << std::endl;
        std::cout << "m_t_hat " << m_t_hat << std::endl;
        std::cout << "v_t_hat " << v_t_hat << std::endl;
        std::cout << "param " << param << std::endl;
        std::cout << "====================" << std::endl;
    }
}
```

这里假设只有一个参数 `param` ，并且梯度 `grad` 为一个 `正` 的常数 `1e-05`，此时输出为：

``` shell
step 1
mu_t 0.450073
mu_t_1 0.450147
mu_product_t 0.450073
mu_product_t_1 0.202599
m_t 1e-06
v_t 9.99987e-14
m_t_hat 1.05645e-05
v_t_hat 1e-10
param 0.121945
====================
step 2
mu_t 0.450147
mu_t_1 0.45022
mu_product_t 0.202599
mu_product_t_1 0.0912143
m_t 1.9e-06
v_t 1.99897e-13
m_t_hat 7.83684e-06
v_t_hat 1e-10
param 0.121162
====================
step 3
mu_t 0.45022
mu_t_1 0.450294
mu_product_t 0.0912143
mu_product_t_1 0.0410732
m_t 2.71e-06
v_t 2.99696e-13
m_t_hat 7.32217e-06
v_t_hat 1e-10
param 0.12043
====================
step 4
mu_t 0.450294
mu_t_1 0.450367
mu_product_t 0.0410732
mu_product_t_1 0.018498
m_t 3.439e-06
v_t 3.99395e-13
m_t_hat 7.31052e-06
v_t_hat 1e-10
param 0.1197
====================
```

可以看到，`param` 不断更新，并且，由于梯度为 `正` 且为常数，因此不断 `减小` 。


## API实现方案
参考其他 Paddle 中优化器的实现方案，此处需要实现：

- python 端 API `class paddle.optimizer.NAdam`
- ops.yaml 的算子配置
- c++ 端算子，CPU 与 GPU 算子
- NAdamInferMeta 推理 dtype 与 shape
- 单元测试
- 文档

# 六、测试和验收的考量
测试考虑的case如下：

- **编程范式场景**
  常规覆盖动态图和静态图的测试场景

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **输出正确性**
  输出数值结果的一致性和数据类型是否正确，使用 PyTorch 作为参考标准

- **计算精度**
  需要保证 `前向/后向` 计算的精度正确性，使用 PyTorch 作为参考标准


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

[2] [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)

[3] [NADAM](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)

[4] [tf.keras.optimizers.Nadam](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Nadam)

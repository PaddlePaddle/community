# PoissonNLLLoss 设计文档

| API名称                                                      | PoissonNLLoss                                |
| ------------------------------------------------------------ |------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | LyndonKong                                      |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-01                                     |
| 版本号                                                       | V1.0                                           |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                        |
| 文件名                                                       | 20230301_api_design_for_poissonllloss.md<br> |


# 一、概述

## 1、相关背景

paddle.nn.PoissonNLLLoss 和 paddle.nn.functional.Poisson_nll_loss API 用于计算真实标签服从泊松分布的负对数似然损失。
该函数计算公式为：

$$
\text{loss}(\text{input}, \text{label}) = \text{input} - \text{label} * \log(\text{input}) + \log(\text{label!})
$$

损失函数中的最后一项可以使用Stirling公式近似
$$
\text{label}*\log(\text{label}) - \text{label} + 0.5 * \log(2\pi\text{label})
$$
将label和每个元素都为1的同样形状的张量比较，对label值超过1的索引处考虑此项近似，对label的值小于等于1的索引处设置此项近似为0进行遮盖。

## 2、功能目标

在飞桨中增加 paddle.nn.PoissonNLLLoss 和 paddle.nn.functional.Poisson_nll_loss API。

## 3、意义

飞桨将支持 paddle.nn.PoissonNLLLoss 和 paddle.nn.functional.Poisson_nll_loss API。

# 二、飞桨现状

飞桨中还没有 PoissonNLLLoss API，可以简单通过log，exp等函数构造该函数。


# 三、业内方案调研

PyTorch：PyTorch 支持 torch.nn.poissonNLLLoss 和 torch.nn.functional.Poisson_nll_loss，由python代码提供接口：

```python
def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    epsilon: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            poisson_nll_loss,
            (input, target),
            input,
            target,
            log_input=log_input,
            full=full,
            size_average=size_average,
            epsilon=epsilon,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        ret = input
        raise ValueError(reduction + " is not valid")

    ret = torch.poisson_nll_loss(input, target, log_input, full, epsilon, _Reduction.get_enum(reduction))
    return ret
```

由cpp提供具体实现
```cpp
    Tensor poisson_nll_loss(const Tensor& input, const Tensor& target, const bool log_input, const bool full, const double epsilon, const int64_t reduction)
    {
        Tensor loss;
        if (log_input) {
            loss = at::exp(input) - target * input;
        } else {
            loss = input - target * at::log(input + epsilon);
        }

        if (full) {
            auto stirling_term = target * at::log(target) - target + 0.5 * at::log(2 * c10::pi<double> * target);
            loss += stirling_term.masked_fill(target <= 1, 0);
        }

        return apply_loss_reduction(loss, reduction);
    }

    static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
        if (reduction == at::Reduction::Mean) {
        return unreduced.mean();
        } else if (reduction == at::Reduction::Sum) {
        return unreduced.sum();
        }
        return unreduced;
    }
```

无其它相关库支持该 Loss 函数。

# 四、对比分析

设计方案将参考PyTorch的实现，在当前版本的API当中仅实现Python版本。由于PyTroch实现当中参数 ``size_average`` 和 ``reduce`` 的功能因为被 ``reduction``包含而废弃，在此版本的API实现我们中对这些被抛弃的参数不再做兼容。

# 五、设计思路与实现方案

## 命名与参数设计

共添加以下两个 API：

`paddle.nn.functional.poisson_nll_loss(
    input,
    label,
    log_input = True,
    full = False,
    epsilon = 1e-8,
    reduction = "mean",
    name:str = None
) -> Tensor:`
 - Input(Tensor): 期望服从泊松分布的输入，形状为`(N, *)` 或 `(*)` 其中 `*`表示任何数量的额外维度。
 - label(Tensor): 为泊松分布的随机样本，形状为`(N, *)` 或 `(*)`，与输入的形状相同， 
或与输入的形状相同但有一个维度等于1（允许广播）。
 - Log_input(bool): 输入和目标是否为对数。如果为`True`，则损失函数的前两项的计算方式为$\exp(\text{input}) - \exp\text{label} * \text{label}$。如果设置为`False`，则损失函数的前两项计算方式为$\text{input} - \text{label} * \log(\text{input}+\text{epsilon})$。默认为`True`。
 - Full(bool)：是否计算完整的损失。如果为`True`，则添加Stirling逼近项$\text{label}*\log(\text{label}) - \text{label} + 0.5 * \log(2\pi\text{label})$。将label和每个元素都为1的同样形状的张量比较，对label值超过1的索引处考虑此项近似，对label的值小于等于1的索引处设置此项近似为0进行遮盖。默认为`False`。
 - epsilon: 避免在`log_input=False`时计算$\log(0)$的小量。默认值为1e-8/。
 - Reduction：指定应用于输出结果的计算方式，指定应用于输出结果的计算方式，可选值有："none", "mean", "sum"。默认为"mean"，计算`Poisson_nll_loss`的均值；设置为"sum"时，计算`Poisson_nll_loss`的总和；设置为"none"时，则返回`Poisson_nll_loss`。
 - Name: 操作的名称，默认为None。

和

`paddle.nn.PoissonNLLLoss(
    log_input = True,
    full = False,
    epsilon = 1e-8,
    reduction = "mean",
    name = None
) -> Tensor:`
- log_input(bool): 输入和目标是否为对数。如果为`True`，则损失函数的前两项的计算方式为$\exp(\text{input}) - \exp\text{label} * \text{label}$。如果设置为`False`，则损失函数的前两项计算方式为$\text{input} - \text{label} * \log(\text{input}+\text{epsilon})$。默认为`True`。
 - Full(bool)：是否计算完整的损失。如果为`True`，则添加Stirling逼近项$\text{label}*\log(\text{label}) - \text{label} + 0.5 * \log(2\pi\text{label})$。将label和每个元素都为1的同样形状的张量比较，对label值超过1的索引处考虑此项近似，对label的值小于等于1的索引处设置此项近似为0进行遮盖。默认为`False`。
 - epsilon: 避免在`log_input=False`时计算$\log(0)$的小量。默认值为1e-8/。
 - Reduction：指定应用于输出结果的计算方式，指定应用于输出结果的计算方式，可选值有："none", "mean", "sum"。默认为"mean"，计算`Poisson_nll_loss`的均值；设置为"sum"时，计算`Poisson_nll_loss`的总和；设置为"none"时，则返回`Poisson_nll_loss`。
 - Name: 操作的名称，默认为None。

参数与文档要求进行对齐。

## API实现方案

参考 pytorch 的处理方式通过 paddle.exp, paddle.log , paddle.where函数实现。
1. 检查参数

   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入参数 epsilon 是否为正数
   3. 检查输入（含 `input`、`label`）的size和dtype（同其余 functional loss 中的实现）

2. 计算

   1. 判断`log_input`是否为`True`，计算loss的前两项
   2. 判断`full`是否为`True`，计算Stirling逼近项
   3. 计算loss


3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）

# 六、测试和验收的考量

由于在numpy当中没有Poisson_nll_loss的实现，我们基于numpy自己实现了此函数，并于参考方案对比验证了numpy实现的正确性。在此基础上我们进行了前向的测验和验收：
1. 结果正确性:

   - 前向计算:`paddle.nn.PoissonNLLLoss` 和 `paddle.nn.functional.poisson_nll_loss` 计算结果与numpy实现计算结果一致。
   - 反向计算:由 Python API 组合新增 API 无需验证反向计算。

2. 硬件场景: 在 CPU 和 GPU 硬件条件下的运行结果一致。

3. 异常测试:

   - 数据类型检验:
     - input和label的数据类型检验
     - 可选参数的数据类型检验
   - 具体数值检验:
     - input 与 label 的维度一致检查
     - 若 epsilon 有输入, 则要为正


# 七、可行性分析和排期规划
方案主要依赖现有 paddle API 组合，待该设计文档通过验收后可尽快提交。

# 八、影响面

在 paddle.nn.functional.loss 文件中import math，新增的API对其他模块没有影响

# 名词解释

# 附件及参考资料

[torch实现](https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#poisson_nll_loss)
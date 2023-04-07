# GaussianNLLLoss 设计文档

| API名称                                                      | GaussianNLLLoss                                |
| ------------------------------------------------------------ |------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | 猫猫教没落                                          |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-22                                     |
| 版本号                                                       | V1.0                                           |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                        |
| 文件名                                                       | 20230221_api_design_for_gaussiannllloss.md<br> |


# 一、概述

## 1、相关背景

paddle.nn.GaussianNLLLoss 和 paddle.nn.functional.gaussian_nll_loss API 用于高斯负对数似然函数的计算。
该函数计算公式为：
$$
\text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{epsilon}\right)\right) + \frac{\left(\text{input} - \text{label}\right)^2}
        {\text{max}\left(\text{var}, \ \text{epsilon}\right)}\right) + \text{const.}
$$

## 2、功能目标

在飞桨中增加 paddle.nn.GaussianNLLLoss 和 paddle.nn.functional.gaussian_nll_loss API。

## 3、意义

飞桨将支持 paddle.nn.GaussianNLLLoss 和 paddle.nn.functional.gaussian_nll_loss API。

# 二、飞桨现状

飞桨中还没有 GaussianNLLLoss API，可以简单通过log，clip等函数构造该函数。


# 三、业内方案调研

PyTorch：PyTorch 支持 torch.nn.GaussianNLLLoss 和 torch.nn.functional.gaussian_nll_loss，也是由python代码实现如下：

```python
def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    if has_torch_function_variadic(input, target, var):
        return handle_torch_function(
            gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
```

无其它相关库支持该 Loss 函数。

# 四、对比分析

可直接采用 PyTorch 的思路转换为 Paddle 实现。

# 五、设计思路与实现方案

## 命名与参数设计

共添加以下两个 API：

`paddle.nn.functional.gaussian_nll_loss(input,
    label,
    variance,
    full=False,
    epsilon=1e-6,
    reduction: str="mean",
    name:str=None,
) -> Tensor:`
 - Input(Tensor): 期望服从高斯分布的输入，形状为`(N, *)` 或 `(*)` 其中 `*`表示任何数量的额外维度。
 - Label(Tensor):为高斯分布的采样值，形状为`(N, *)` 或 `(*)`，与输入的形状相同， 
或与输入的形状相同但有一个维度等于1（允许广播）。
 - Variance(Tensor): 正方差张量，即数值均大于等于0的方差张量，形状为`(N, *)` 或 `(*)`，与输入的形状相同，或与输入的形状相同但有
一个维度等于1，或与输入的形状相同但少一个维度（允许广播）。
 - Output(Tensor): 输出衡量Input与Label差距的损失函数结果，如果‘reduction’是 “mean”（默认）或 “sum”，则为标量。如果‘reduction’是’none’，
则是`(N, *)`，与输入的形状相同。

和

`paddle.nn.GaussianNLLLoss(full,
    epsilon,
    reduction,
    name) -> Tensor:`
- full(bool):默认为False
- epsilon(float):默认为1e-6
- reduction(Optional|str): 可选项：'None' , 'mean' , 'sum', 默认为'mean'
- name(str):

参数与文档要求进行对齐。

## API实现方案

参考 pytorch 的处理方式通过 paddle.clip, paddle.log 函数实现。
1. 检查参数
  
   1. 检查 reduction 有效性（同其余 functional loss 中的实现）
   2. 检查输入的 size（含 `input`、`label`、`variance`）（同其余 functional loss 中的实现）
   3. 检查输入的`input`、`label`、`weight`是否可以广播

2. 计算

   1. 判断variance是否小于epsilon
   2. 计算loss


3. 根据 `reduction`，输出 loss（同其余 functional loss 中的实现）


# 六、测试和验收的考量

numpy没有gaussian_nll_loss的实现，所以由自己生成的函数进行前向测试和验收：

1. 测试 API 在动态图和静态图下与 numpy 的一致性。
2. 测试 CPU、GPU 上与 numpy 的一致性。
3. 各reduction下计算一致性。
4. 测试 `float32` 数据类型与 numpy 的一致性。
5. 各参数输入有效。

# 七、可行性分析和排期规划
函数均为python代码实现，已经基本实现，待该设计文档通过验收后可在短时间内提交。

# 八、影响面

在paddle.nn.functional.loss 文件中import math

# 名词解释

# 附件及参考资料

[torch实现](https://pytorch.org/docs/stable/generated/torch.nn.functional.gaussian_nll_loss.html?highlight=gaussiannllloss)
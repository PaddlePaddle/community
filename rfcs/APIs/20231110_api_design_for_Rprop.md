#  Rprop API 设计文档

| API 名称 | Rprop |
| - | - |
| 提交作者 | WintersMontagne10335 |
| 提交时间 | 2023-11-10 |
| 版本号 | V1.0 |
| 依赖飞桨版本 | develop |
| 文件名 | 20231110_api_design_for_Rprop.md |

# 一、概述

## 1、相关背景

常规反向传播算法存在两个问题：
- 对于不同的权值参数，很难选到一个适用于全局的学习率
- 反向传播算法的梯度具有弥散作用，即距离输出层越远的神经元学习的速度越慢

弹性反向传播算法（下面简称 `Rprop`）由此被提出：
- 在 `Rprop` 中，每一个可优化的权重，都对应着一个单独的学习率，这些学习率在程序执行过程中不断地更新
- 误差函数梯度，并不通过值直接作用于权重值的变化，而是通过符号及符号的变化影响步长，进而间接地影响权重值的变化

## 2、功能目标

新增 `Rprop` API。

调用形式：
- `paddle.optimizer.Rprop`

## 3、意义

为 `Paddle` 增加 `Rprop` ，丰富 `Paddle` 中优化器相关的 API。

# 二、飞桨现状

`Paddle` 目前已经提供了 `SGD` 等优化器方法。

目前 `Paddle` 在 `Python` 端缺少 `Rprop` 相关接口的实现，而在底层也没有相关算子。

# 三、业内方案调研

## PyTorch

`Pytorch` 底层并未实现 `Rprop` 直接对应的 `Kernel`，而是通过在 `Python` 端，基于 `foreach` 系列，组合实现了 API。

### API 文档

- [torch.optim.Rprop(params, lr, etas, step_sizes, *, foreach, maximize, differentiable)](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#rprop)

    - sketch
        - Implements the resilient backpropagation algorithm

    - Parameters
        - params
        > iterable of parameters to optimize or dicts defining parameter groups
        - lr
        > learning rate
        - etas
        > pair of (etaminus, etaplus), that are multiplicative increase and decrease factors
        - step_sizes
        > a pair of minimal and maximal allowed step sizes
        - foreach
        > whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more performant. Note that the foreach implementation uses ~ sizeof(params) more peak memory than the for-loop version due to the intermediates being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer parameters through the optimizer at a time or switch this flag to False
        - maximize
        > maximize the params based on the objective, instead of minimizing
        - differentiable
        > whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you don’t intend to run autograd through this instance

### 实现逻辑 

#### `Python` 端

关键源码

- [pytorch/torch/optim/rprop.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/rprop.py)

```Python
    def __init__(
        self,
        params,
        lr=1e-2,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50),
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)
```

检验lr、etas的正确性。初始化。

```Python
    def _init_group(self, group, params, grads, prevs, step_sizes):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Rprop does not support sparse gradients")

            grads.append(grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                state["prev"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if p.dtype.is_complex:
                    # Complex Number should be as if they are two independent real numbers.
                    # Hence the step_size shouldn't be zero for imaginary part.
                    state["step_size"] = (
                        grad.new()
                        .resize_as_(grad)
                        .fill_(complex(group["lr"], group["lr"]))
                    )
                else:
                    state["step_size"] = (
                        grad.new().resize_as_(grad).fill_(group["lr"])
                    )

            prevs.append(state["prev"])
            step_sizes.append(state["step_size"])

            state["step"] += 1
        return has_complex
```

初始化 `params`、 `grads`、 `prevs`、 `step_sizes`。这里的 `prevs` 为上次的梯度。

```Python
def rprop(
    params: List[Tensor],
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
):
    r"""Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop

    func(
        params,
        grads,
        prevs,
        step_sizes,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        etaminus=etaminus,
        etaplus=etaplus,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
    )
```

根据 `foreach` 与 `torch.jit.is_scripting()` 确定要执行的函数。 `_single_tensor_rprop` 为纯 `Python` 实现， `_multi_tensor_rprop` 使用了 `foreach` 系列的算子，有加速效果。

```Python
def _single_tensor_rprop(
    params: List[Tensor],
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        prev = prevs[i]
        step_size = step_sizes[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grad = grad.clone(memory_format=torch.preserve_format)
        grad[sign.eq(etaminus)] = 0

        # update parameters
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)
```

这里与原论文的实现略有不同，原论文中，当上次的梯度符号与本次梯度符号相反时，会对权重值做一个“回溯”的操作，即

TODO:LaTeX公式

`PyTorch` 无此操作。

#### CPU端

`PyTorch` 未实现。

#### GPU端

`PyTorch` 未实现。

## TensorFlow

`TensorFlow` 未实现该算子。

## MXNet

`MXNet` 未实现该算子。

## OneFlow

`OneFlow` 未实现该算子。

# 四、对比分析

以下均为 `PyTorch` 的 `Rprop` 与其它的对比。

- 与原论文的实现相比：在上次的梯度符号与本次梯度符号相反时，原论文会“回溯”权重值；`PyTorch` 不会
- 与 `Paddle` 的 `SGD` 相比： `SGD` 有直接对应的底层实现； `Rprop` 则没有，而且在 `Python` 端，用 `foreach` 系列组合实现了该 `API`。

# 五、设计思路与实现方案

## 命名与参数设计

添加 python 上层接口:

- `paddle.optimizer.Rprop`

    ``` python
    paddle.optimizer.Rprop(
        delta = 0.01,
        delta_min = 1e-6,
        delta_max = 50,
        parameters = None,
        eta_negative = 0.5,
        eta_positive = 1.2,
        grad_clip = None,
        name = None
    ):
    ```

    |参数名|类型|描述|
    |---|---|---|
    |delta|float|used to update ``Parameter``|
    |delta_min|float|delta cannot be smaller than delta_min|
    |delta_max|float|delta cannot be larger than delta_max|
    |parameters|list, tuple|list / tuple of ``Tensor`` to update to minimize ``loss``|
    |eta_negative|float|multiplicative decrease factor|
    |eta_positive|float|multiplicative increase factor|
    |grad_clip|GradientClipBase|gradient cliping strategy|
    |name|str|normally there is no need for user to set this property|

## 底层 OP 设计

TODO

做完底层实现后补充。

## API实现方案

- `paddle.optimizer.Rprop`

TODO

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

# 七、可行性分析及规划排期

11.20 前做出第一版提交审核。

# 八、影响面

新增 API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

- [torch.optim.Rprop](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#rprop)
- [paddle.optimizer.SGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/SGD_cn.html)

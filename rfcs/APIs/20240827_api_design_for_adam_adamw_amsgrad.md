# paddle.optimizer.Adam/paddle.optimizer.AdamW 设计文档

| API名称      | paddle.optimizer.Adam/paddle.optimizer.AdamW                     |
| ------------ | --------------------------------------- |
| 提交作者     | NKNaN                                 |
| 提交时间     | 2024-08-27                              |
| 版本号       | V1.0                                    |
| 依赖飞桨版本 | develop                                 |
| 文件名       | 20240827_api_design_for_adam_adamw_amsgrad.md |

# 一、概述

## 1、相关背景

最近提出的几种随机优化方法如RMSProp、Adam、Adadelta、Nadam，已经成功地用于训练深度网络，这些梯度更新的方法都是使用梯度平方的指数移动平均的平方根来进行缩放的。在许多应用中，例如大输出空间的学习，根据经验观察到这些算法无法收敛到最优解(或非凸设定中的临界点)。[On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)中通过分析指出，这种失败的原因之一是在算法中使用了指数移动平均。论文提供了一个简单的凸优化设定的显式例子，其中Adam不收敛于最优解。其分析表明，该不收敛的问题可以通过赋予这些算法对于过去梯度的“长期记忆”来解决，并提出亚当算法的新变体，不仅可以解决收敛问题，而且通常还可以提高经验性能。

使用 AMSGrad 方法的 Adam 算法：
![adam](\image\img_Adam.png)

使用 AMSGrad 方法的 AdamW 算法：
![adamw](\image\img_AdamW.png)

## 2、功能目标

为 paddle.optimizer.Adam 和 paddle.optimizer.AdamW 添加 amsgrad 支持，添加动态图和静态图的实现，在单机和分布式环境下，精度与 pytorch 对齐。

## 3、意义

完善 Paddle optimizer 中 Adam 和 AdamW 的功能

# 二、飞桨现状

目前 Paddle 缺少相关功能实现。

# 三、业内方案调研

## PyTorch

- PyTorch 中的 Adam（https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam）

- PyTorch 中的 AdamW（https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW）

### AMSGrad 实现方法

- Adam：
```py
def _single_tensor_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(
                    device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
        )

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]  # type: ignore[arg-type]

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(
                device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size  # type: ignore[arg-type]
            )


def _fused_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,  # Needed for consistency.
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict: Optional[DeviceDict] = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if device.type == "mps":  # type: ignore[union-attr]
            assert found_inf is None and grad_scale is None

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adam_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,  # type: ignore[arg-type]
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,  # type: ignore[arg-type]
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adam)
def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adam
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
```

- AdamW:
```py
def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(
                    device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
        )

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [
                bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
            ]

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(
                device_params,
                device_exp_avgs,
                exp_avg_sq_sqrt,
                step_size,  # type: ignore[arg-type]
            )


def _fused_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
    has_complex: bool,  # Needed for consistency.
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict: Optional[DeviceDict] = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if device.type == "mps":  # type: ignore[union-attr]
            assert found_inf is None and grad_scale is None

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        if lr_dict is not None and device not in lr_dict:
            lr = lr_dict.setdefault(
                device, lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            )
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adamw_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,  # type: ignore[arg-type]
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,  # type: ignore[arg-type]
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamw)
def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adamw
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
```

## TensorFlow

- TensorFlow 中的 Adam（https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam）

- TensorFlow 中的 AdamW（https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/AdamW）


### 实现方法

- Adam:
```py
@keras_export(["keras.optimizers.Adam"])
class Adam(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="velocity_hat"
                    )
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(
            ops.cast(self.beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.beta_2, variable.dtype), local_step
        )

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.beta_2
            ),
        )
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)
            ),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
```

- AdamW:
```py
@keras_export(["keras.optimizers.AdamW"])
class AdamW(adam.Adam):
    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )

        if self.weight_decay is None:
            raise ValueError(
                "Argument `weight_decay` must be a float. Received: "
                "weight_decay=None"
            )
```

# 四、对比分析

pytorch 和 tensorflow 的 Adam 和 AdamW 优化器都是用 python API 组合实现，而 paddle 底层调用 c++ kernel，所以具体的 amsgrad 实现方法需要适配于 paddle 框架。


# 五、方案设计

## 命名与参数设计

paddle.optimizer.Adam 和 paddle.optimizer.AdamW 需增加初始化参数 amsgrad(bool, optional) = False，表示是否应用 amsgrad 方法。


## 底层OP设计 - 以 Adam 的 cpu kernel 为例

```cpp
#include "paddle/phi/kernels/elementwise_kernel.h" // new

template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     const paddle::optional<DenseTensor>& master_param,
                     const paddle::optional<DenseTensor>& skip_update,
                     const paddle::optional<DenseTensor>& max_moment2,  //new
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& epsilon,
                     bool lazy_mode,
                     int64_t min_row_size_to_use_multithread,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     DenseTensor* param_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* master_param_outs,
                     paddle::optional<DenseTensor>* max_moment2_out) {  //new
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    phi::TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }
  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    if (max_moment2) {  //new
      phi::Copy(dev_ctx, max_moment2, dev_ctx.GetPlace(), false, max_moment2_out);  //new
    }  //new
    return;
  }

  T beta1_ = beta1.to<T>();
  T beta2_ = beta2.to<T>();
  T epsilon_ = epsilon.to<T>();

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel();
  VLOG(3) << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  T beta1_p = beta1_pow.data<T>()[0];
  T beta2_p = beta2_pow.data<T>()[0];

  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out)[0] = beta1_ * beta1_p;
    dev_ctx.template Alloc<T>(beta2_pow_out)[0] = beta2_ * beta2_p;
  }

  T* param_out_ptr = dev_ctx.template Alloc<T>(param_out);
  T* mom1_out_ptr = dev_ctx.template Alloc<T>(moment1_out);
  T* mom2_out_ptr = dev_ctx.template Alloc<T>(moment2_out);
  T* max_mom2_out_ptr; //new
  if (max_moment2_out) { //new
    max_mom2_out_ptr = dev_ctx.template Alloc<T>(max_moment2_out); //new
  } //new

  T learning_rate_ =
      learning_rate.data<T>()[0] * (sqrt(1 - beta2_p) / (1 - beta1_p));
  T eps = epsilon_ * sqrt(1 - beta2_p);

  phi::jit::adam_attr_t attr(beta1_, beta2_);
  int64_t numel = param.numel();

  const T* param_ptr = param.data<T>();
  const T* mom1_ptr = moment1.data<T>();
  const T* mom2_ptr = moment2.data<T>();
  const T* grad_ptr = grad.data<T>();
  const T* max_mom2_ptr; //new
  if (max_moment2) { //new
    max_mom2_ptr = max_moment2.data<T>(); //new
  } //new

  auto adam =
      phi::jit::KernelFuncs<phi::jit::AdamTuple<T>, phi::CPUPlace>::Cache().At(
          attr);

  static constexpr int64_t chunk_size = 512;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < numel / chunk_size; ++i) {
    const int64_t offset = i * chunk_size;
    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         chunk_size,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         param_ptr + offset,
         max_mom2_ptr + offset,  // new
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         param_out_ptr + offset,
         max_mom2_out_ptr + offset);  // new
  }

  if (numel % chunk_size != 0) {
    const int64_t offset = (numel / chunk_size) * chunk_size;
    const int64_t tail_numel = numel % chunk_size;
    adam(beta1_,
         beta2_,
         -learning_rate_,
         eps,
         tail_numel,
         grad_ptr + offset,
         mom1_ptr + offset,
         mom2_ptr + offset,
         param_ptr + offset,
         max_mom2_ptr + offset,  // new
         mom1_out_ptr + offset,
         mom2_out_ptr + offset,
         param_out_ptr + offset,
         max_mom2_out_ptr + offset);  // new
  }
}

template <typename T>
void Adam(T beta1,
          T beta2,
          T lr,
          T eps,
          int64_t numel,
          const T* grad_ptr,
          const T* mom1_ptr,
          const T* mom2_ptr,
          const T* param_ptr,
          const T* max_mom2_ptr,  // new
          T* mom1_out_ptr,
          T* mom2_out_ptr,
          T* param_out_ptr,
          T* max_mom2_out_ptr) {  // new
  if (max_mom2_ptr) {  // new
    for (int i = 0; i < numel; ++i) {  // new
      mom1_out_ptr[i] = beta1 * mom1_ptr[i] + (1 - beta1) * grad_ptr[i];  // new
      mom2_out_ptr[i] =
          beta2 * mom2_ptr[i] + (1 - beta2) * grad_ptr[i] * grad_ptr[i];  // new
      max_mom2_out_ptr[i] = max(max_mom2_ptr[i], mom2_out_ptr[i]);  // new
      param_out_ptr[i] =
          param_ptr[i] + lr * (mom1_out_ptr[i] / (sqrt(max_mom2_out_ptr[i]) + eps));  // new
    }  // new
  } else {  // new
    for (int i = 0; i < numel; ++i) {
      mom1_out_ptr[i] = beta1 * mom1_ptr[i] + (1 - beta1) * grad_ptr[i];
      mom2_out_ptr[i] =
          beta2 * mom2_ptr[i] + (1 - beta2) * grad_ptr[i] * grad_ptr[i];
      param_out_ptr[i] =
          param_ptr[i] + lr * (mom1_out_ptr[i] / (sqrt(mom2_out_ptr[i]) + eps));
    }
  }
}
```

## API实现方案 - 以 Adam 为例

```python
class Adam(Optimizer):
    type: str
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"
    _max_moment2_acc_str = "max_moment2"  # new

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
        amsgrad: bool = False,  # new
        name: str | None = None,
    ) -> None:
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not isinstance(beta1, (Variable, Value)):
            if not 0 <= beta1 < 1:
                raise ValueError(
                    "Invalid value of beta1, expect beta1 in [0,1)."
                )
        if not isinstance(beta2, (Variable, Value)):
            if not 0 <= beta2 < 1:
                raise ValueError(
                    "Invalid value of beta2, expect beta2 in [0,1)."
                )
        if not isinstance(epsilon, (Variable, Value)):
            if not 0 <= epsilon:
                raise ValueError(
                    "Invalid value of epsilon, expect epsilon >= 0."
                )
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._multi_precision = multi_precision
        self._amsgrad = amsgrad  # new
        self._master_weights = {}
        self._default_dict = {
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            'lazy_mode': lazy_mode,
        }

        self._use_multi_tensor = use_multi_tensor
        if self._use_multi_tensor:
            self._param_dict = self._create_multi_tensor_dict()
            self._moment1_dict = self._create_multi_tensor_dict()
            self._moment2_dict = self._create_multi_tensor_dict()
            self._beta1_pow_acc_dict = self._create_multi_tensor_dict()
            self._beta2_pow_acc_dict = self._create_multi_tensor_dict()
            self._max_moment2_dict = self._create_multi_tensor_dict() if self._amsgrad else None   # new
            self._master_weight_dict = self._create_multi_tensor_dict()
            self._master_weight_dict['FP32_LODTensor'] = None

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            if in_pir_mode():
                acc_dtype = DataType.FLOAT32
            else:
                acc_dtype = core.VarDesc.VarType.FP32
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        if self._amsgrad:  # new
          self._add_accumulator(self._max_moment2_acc_str, p, dtype=acc_dtype)  # new
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(
                0.9
                if isinstance(self._beta1, (Variable, Value))
                else self._beta1
            ),
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR,
            device='cpu',
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(
                0.999
                if isinstance(self._beta2, (Variable, Value))
                else self._beta2
            ),
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR,
            device='cpu',
        )

    ...
```


# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：与 pytorch 的精度对齐；
  - 单机和分布式环境；
  - 动态图静态图；

- 不同计算设备：覆盖 CPU 和 GPU 等实现；

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

# 八、影响面

对其他模块没有影响

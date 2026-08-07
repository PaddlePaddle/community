# 修复 `is_in_recompute()` 在反向重计算时返回错误的问题

## 详细描述

在动态图中使用 `recompute(..., use_reentrant=True)` 时，传入的计算函数会执行两次：第一次是正常的前向计算，第二次是反向传播时的重新计算。

`is_in_recompute()` 在第一次执行时返回 `True`，但在反向重计算时返回 `False`，导致依赖该接口判断 recompute 状态的代码在反向阶段走错分支。

计算函数在这两次执行期间都应处于 recompute 状态。执行结束后，`is_in_recompute()` 应恢复为 `False`。

## 验收说明

- 正常前向计算期间，`is_in_recompute()` 应返回 `True`
- 反向传播触发重新计算时，`is_in_recompute()` 也应返回 `True`
- `preserve_rng_state=True` 和 `False` 时都应正确
- recompute 执行结束后，`is_in_recompute()` 应返回 `False`
- 现有的计算结果和梯度应保持不变

## 技术要求

- 熟悉 Python
- 了解 Paddle 动态图和 `reompute`
- 了解自动微分与反向重计算
- 了解 Python 上下文管理器

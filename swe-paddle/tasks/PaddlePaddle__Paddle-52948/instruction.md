# 在静态图与动转静场景下支持 Tensor.register_hook

## 详细描述

当前动态图中可以对 Tensor 调用 `register_hook` 注册反向 hook，并在 `backward` 时按预期改写梯度。但在静态图，以及经 `paddle.jit.to_static`（或等价装饰器）转换后的路径上，该能力不可用或行为与动态图不一致。

需要达成的目标：

- 在静态图模式下，对可求导的 Tensor / Variable 调用 `register_hook` 应能成功执行，而不再因接口不可用而直接失败。
- 对经过 `to_static` 转换的函数或 `nn.Layer`，在中间结果或参数上注册的反向 hook 应能被触发。
- 同一计算在动态图与 `to_static` 路径下，hook 对梯度的影响应保持一致。至少覆盖以下场景：
  - 多个中间变量分别注册 hook
  - 变量被重新赋值后再注册 hook
  - 同一变量重复注册 hook
  - 在 `nn.Layer.__init__` 中对参数注册 hook，再对 `to_static` 后的网络做前向与反向
- 不要求支持：嵌套内部函数中的 `register_hook`，以及 `hook.remove`。

## 问题复现（示意）

在 `base_commit` 对应版本上：

1. 开启静态图，构造简单网络并对中间结果调用 `register_hook`，或对 `@to_static` / `to_static(...)` 包装后的函数/Layer 注册 hook 再 `backward`。
2. 可观察到接口不可用（例如断言失败），或 hook 未按动态图语义影响梯度。
3. 对照同一计算的纯动态图路径，梯度行为不一致或静态 / 动转静路径无法完成。

## 期望行为

- 静态图与动转静路径下 `register_hook` 可运行。
- hook 触发后的梯度结果与动态图一致（在上述覆盖场景内）。
- 既有动态图 `register_hook` 行为保持不变。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有动态图 hook 语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python
- 了解 Paddle 动态图 / 静态图与动转静（`to_static`）基本机制
- 了解 autograd hook 的基本语义

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

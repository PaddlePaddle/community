# 为 Tensor / Variable 新增 apply / apply_ API

## 详细描述

需要为 Paddle Tensor（及相关 Variable 路径）新增逐元素 `apply` / `apply_` 能力：用户传入可调用对象，对 Tensor 元素应用自定义变换，并得到 out-of-place 或 inplace 结果。

需要达成的目标：

- 提供 `apply`：返回新 Tensor，元素为对原元素应用可调用对象后的结果。
- 提供 `apply_`：inplace 修改原 Tensor，并返回同一对象。
- 动态图主路径可用；相关静态 / PIR 路径上的基本可用性与错误处理应符合既有约定（例如对需要梯度或不合法可调用对象给出明确失败）。
- 既有其他 Tensor / inplace API 不被破坏。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 构造简单 Tensor，调用 `x.apply(fn)` 或 `x.apply_(fn)`。
2. 可观察到接口不存在，或行为不符合逐元素应用语义。

## 期望行为

- `apply` / `apply_` 可按上述语义运行。
- inplace 与非 inplace 在覆盖场景内结果一致，且 inplace 保持对象身份。
- 非法用法应失败而非静默给错结果。
- 既有无关 Tensor API 行为保持不变。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有无关语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle Tensor 方法挂载与 pybind / eager 绑定机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

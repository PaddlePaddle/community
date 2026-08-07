# 为 Paddle 新增 masked_scatter API

## 详细描述

Paddle 需要提供与常见 masked scatter 语义对齐的公开 API，用于按布尔 mask 将 `value` 中的元素依次写入目标 Tensor 对应位置。

需要达成的目标：

- 提供非 inplace 接口 `masked_scatter`：在 `mask` 为 True 的位置，按顺序从 `value` 取值写入结果；其余位置保持原输入。
- 提供 inplace 接口 `masked_scatter_`：语义与非 inplace 一致，但就地更新输入 Tensor。
- `mask` 应作为布尔掩码使用；`value` 的元素数量需足以覆盖 `mask` 中为 True 的位置。
- 支持常见数值 dtype，以及动态图与静态图基本用法；对广播后的 `mask` 形状场景保持正确。
- 对非法 dtype / mask / 元素数量不足等情况给出明确错误，而不是静默给出错误结果。
- 既有其他 tensor manipulation / inplace API 行为不被破坏。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 尝试调用 `paddle.masked_scatter` / `Tensor.masked_scatter`（及 inplace 变体）。
2. 可观察到 API 不存在，或无法完成按 mask 散射写入。
3. 即使存在初步实现，在部分 mask 类型或静态图边界场景下也可能出现不稳定失败或错误拦截。

## 期望行为

- `masked_scatter` / `masked_scatter_` 可用，且结果符合 masked scatter 语义。
- 动态图与静态图基本路径可运行。
- 非法输入被正确拒绝。
- 既有无关 API 行为保持不变。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python
- 了解 Paddle Tensor API 与基本动态图 / 静态图用法

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

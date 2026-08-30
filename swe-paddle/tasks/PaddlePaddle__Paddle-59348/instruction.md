# 在 PIR 路径下补齐 sequence_mask 算子支持

## 详细描述

`sequence_mask` 在旧执行路径可用，但在 PIR 路径下支持不完整，相关 sequence / 动转静用例无法在开启 PIR 时正确运行。

需要达成的目标：

- 在 PIR 路径下支持 `sequence_mask`，使开启 PIR executor 后相关用例可运行。
- 输出 mask 的形状与数值语义应与既有非 PIR 行为一致（在任务覆盖范围内）。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 在 `FLAGS_enable_pir_in_executor=true`（或等价 PIR 覆盖方式）下运行 `sequence_mask` 相关用例。
2. 可观察到算子缺失、infermeta / kernel 不完整，或结果不符合预期。

## 期望行为

- PIR 下 `sequence_mask` 可运行且语义正确。
- 既有无关 sequence / 动转静行为不被破坏。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有无关语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle 算子 YAML、infermeta 与 CPU/GPU kernel 相关机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

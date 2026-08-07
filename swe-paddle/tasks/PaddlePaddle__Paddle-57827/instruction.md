# 在 PIR 路径下补齐 fused_elemwise_add_activation 支持

## 详细描述

动转静 / build strategy 场景中可能用到 `fused_elemwise_add_activation` 融合算子。当前在 PIR 路径下该融合算子支持不完整，相关用例无法在 PIR 下正确运行。

需要达成的目标：

- 在 PIR 路径下支持 `fused_elemwise_add_activation`（含前向与反向所需能力）。
- 开启 PIR 后，相关 build strategy / ResNet 类动转静用例应能完成训练或校验路径，并与非 PIR 结果在任务覆盖范围内一致。
- 不要求一次迁移全部无关 fusion 算子。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 运行依赖该融合能力的动转静 build strategy 用例，并开启 PIR 对比路径。
2. 可观察到算子缺失、翻译 / 适配失败，或 PIR 路径无法完成。

## 期望行为

- PIR 下该 fused op 相关路径可运行。
- 任务覆盖范围内的结果与既有路径一致。
- 既有无关 build strategy 行为不被破坏。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有无关动转静 / build strategy 语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle fusion infermeta、PIR 算子配置与图翻译 / 适配机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

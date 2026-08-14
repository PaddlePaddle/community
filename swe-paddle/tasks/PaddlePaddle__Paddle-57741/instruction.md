# 在 PIR 路径下补齐 memcpy 算子支持

## 详细描述

在动转静（`to_static`）场景中，Tensor 在不同 Place 之间拷贝（例如 CPU ↔ CUDA）依赖框架内部的 `memcpy` 能力。当前在 PIR 执行路径下，该能力缺失或不完整，导致相关拷贝用例无法在 PIR 下正确运行。

需要达成的目标：

- 在 PIR 路径下支持 `memcpy`，使动转静场景中的 Tensor 设备间拷贝可以完成。
- 至少覆盖默认在 CPU 上的拷贝相关用例；若环境具备 CUDA，相关 GPU 用例也应可走通。
- 拷贝后的 place 与数值结果应与非 PIR / 既有路径一致。
- 不要求扩展到其他无关的设备管理或拷贝 API。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 构造简单动转静函数，对 Tensor 执行跨 Place 拷贝（例如拷到 CPU / CUDA）。
2. 在 PIR 对比 / 开启路径下运行上述用例。
3. 可观察到算子缺失、lowering 失败，或拷贝结果 / place 不符合预期。

## 期望行为

- PIR 下 `memcpy` 相关路径可运行。
- 拷贝后的 place 与数值与既有非 PIR 行为一致（在任务覆盖范围内）。
- 既有无关动转静行为不被破坏。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有无关动转静语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle PIR 算子声明、兼容配置与 kernel lowering 相关机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

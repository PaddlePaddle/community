# 为若干激活 API 补齐 inplace 能力

## 详细描述

部分 `paddle.nn` / `paddle.nn.functional` 激活接口在动态图中缺少可用的 inplace 路径，或 inplace 与非 inplace 在相关执行路径上行为不一致。需要为以下激活补齐 inplace 能力，并使动态图与相关静态 / 符号形状路径保持一致且可测：

- `CELU`
- `RReLU`
- `Swish`
- `Mish`
- `HardSigmoid`
- `SELU`

需要达成的目标：

- 上述激活的 functional API 与对应 `nn.Layer` 包装均支持 inplace 用法。
- 同一输入下，inplace 与非 inplace 的前向结果（以及需要覆盖的梯度行为）保持一致。
- 相关静态图 / 符号形状推断路径对上述激活的 inplace 形式可用，不因缺失支持而失败。
- 既有非 inplace 行为、数值语义与公开调用方式不被破坏。
- 不要求一次覆盖全部 `paddle.nn.*` 激活，仅聚焦上述列表。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 对上述激活构造简单输入，分别走非 inplace 与 inplace（或期望的 inplace 开关 / 调用形式）。
2. 可观察到 inplace 路径不可用、接口报错，或与非 inplace 结果不一致。
3. 在需要符号形状推断的静态相关路径上，inplace 形式可能缺少支持或无法完成覆盖。

## 期望行为

- 上述激活的 inplace 路径可运行。
- inplace 与非 inplace 结果一致（在任务覆盖范围内）。
- 相关符号形状路径可用。
- 既有非 inplace 行为保持不变。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有非 inplace 激活语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle 激活 API、算子配置与 PIR / 符号形状相关机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

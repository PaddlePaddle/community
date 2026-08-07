# 修复 PIR 下 SelectedRows 相关 len 路径并开放对应 PIR 覆盖测试

## 详细描述

在 PIR executor 路径下，对 `SelectedRows` 相关变量执行 `len` / shape 等访问时可能失败或行为不正确；同时部分 fuse elewise add activation 相关用例尚未在 PIR 覆盖门禁中打开。

需要达成的目标：

- PIR 路径下，对 `SelectedRows` 及其取出的 dense tensor 做 `len` 时结果正确、可运行。
- 相关 fuse elewise add activation 用例可在开启 PIR executor 的条件下被纳入验证并通过。
- 既有非目标 `len` / SelectedRows / pass 行为不被破坏。

## 问题复现（示意）

在目标 `base_commit` 对应版本上：

1. 构造含 `SelectedRows` 的静态程序，在 PIR executor 下执行 `len` 相关逻辑。
2. 或在 `FLAGS_enable_pir_in_executor=true` 下运行 fuse elewise add activation 相关用例。
3. 可观察到类型未支持、shape / len 结果错误，或 PIR 覆盖用例失败。

## 期望行为

- PIR 下 SelectedRows 相关 `len` 路径可运行且结果正确。
- 对应 PIR 覆盖用例可通过。
- 既有无关用例继续通过。

## 验收说明

- 上述目标行为可用，相关用例通过。
- 已有无关语义不被破坏。
- 不允许通过删除测试、弱化断言或大范围绕过校验来「通过」任务。

## 技术要求

- 熟悉 Python 与 C++
- 了解 Paddle PIR adaptor、SelectedRows 与 executor 相关机制

## Acceptance Criteria

- The behavior described above should be implemented.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

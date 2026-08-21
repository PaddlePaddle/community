# 修复 `paddle.iinfo` 的无符号 64 位边界值报告

## 详细描述

`paddle.iinfo(paddle.uint64)` 应当按照无符号 64 位整数类型的语义向 Python 调用方暴露边界信息。当前异常行为会使 `uint64` 的最大值在 Python 侧表现为有符号解释结果，而不是完整的无符号 64 位最大值。

请修复该 API 的可观察行为，使 `paddle.iinfo(paddle.uint64).max` 返回 `18446744073709551615`，并保持其他整数类型的 `min`、`max`、`bits` 和 `dtype` 等既有行为不变。修复应针对真实运行时暴露的类型边界值问题，而不是通过删除测试、弱化断言或在 Python 测试路径中硬编码返回值来规避。

## 验收说明

- `paddle.iinfo(paddle.uint64).max` 必须等于 `18446744073709551615`。
- `paddle.iinfo(paddle.uint64).max` 的返回值必须是 Python `int`。
- `paddle.iinfo(paddle.uint64).min` 必须保持为 `0`，`bits` 必须保持为 `64`。
- `paddle.iinfo(paddle.int64)` 的有符号边界必须保持为 `-9223372036854775808` 和 `9223372036854775807`。
- `paddle.iinfo(paddle.uint32).max` 必须保持为 `4294967295`。
- 不应改变现有广义 `iinfo` / `finfo` 行为，新增测试应聚焦于明确的 `uint64` 边界回归和相邻整数类型护栏。

## 技术要求

- 熟悉 Paddle Python API 与 native runtime 之间的边界值传递。
- 熟悉 Python 整数类型与无符号 64 位整数边界的兼容性问题。
- 保持修改范围最小，避免引入与字符串别名、额外 dtype 输入形式或无关 API 行为相关的新需求。
- 验证时必须加载包含修复的 Paddle runtime；仅修改源码但继续导入旧的未重建二进制运行时不足以验证该问题。

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.
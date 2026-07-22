# 补齐 PIR OpResult 的常用二元运算符

## 详细描述

在 PIR 静态图中，`paddle.static.data` 等接口返回 `paddle.pir.OpResult`。该对象已有
部分 Tensor 数学方法，但缺少若干常用 Python 二元运算符，导致合法的静态图表达式
在构图阶段直接抛出 `TypeError`；其中整除接口还没有正确识别 PIR 执行模式。

需要让 PIR `OpResult` 支持以下行为：

- `x ** y` 执行逐元素幂运算，且现有的 `x.pow(2)` 调用继续可用。
- `x // y`、`x.floor_divide(y)` 和 `x.__floordiv__(y)` 行为一致。
- `x % y`、`x.mod(y)` 和 `x.__mod__(y)` 行为一致。
- `x @ y`、`x.matmul(y)` 和 `x.__matmul__(y)` 行为一致。
- 上述表达式构造出的 PIR 程序可在 CPU executor 中运行，结果与对应的 NumPy
  或动态图计算一致。

本任务不要求扩展原 PR 未验证的标量反向幂场景。

## 验收说明

- 四类运算均能在 PIR 静态图中完成构图和执行，不再因缺失运算符而报错。
- float32 的幂和矩阵乘法、int64 的整除和取模结果正确。
- 既有的 `item`、`place`、维度属性和其他数学方法保持可用。
- 不允许通过删除测试、弱化断言、切回旧静态图或绕过 PIR executor 来通过任务。

## 技术要求

- 熟悉 Python 运算符协议。
- 了解 Paddle PIR 静态图和 `OpResult` 的公开行为。
- 修复应局限于必要的 Python API 行为，不修改算子数值语义或底层 kernel。

## Acceptance Criteria

- The PIR `OpResult` behavior described above should be implemented.
- Existing valid `OpResult` methods and attributes should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, leaving PIR mode, or bypassing execution.

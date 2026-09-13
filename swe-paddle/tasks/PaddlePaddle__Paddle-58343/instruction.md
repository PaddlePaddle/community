# 补齐 PIR OpResult 的比较、位运算与标量幂行为

## 详细描述

在 PIR 静态图中，`paddle.static.data` 等接口返回 `paddle.pir.OpResult`。该对象的部分 Python 比较和位运算协议尚不完整，相关表达式可能在构图阶段抛出 `TypeError`、退化成 Python `bool`，或进入不支持 PIR 的旧静态图路径；标量幂调用也无法可靠构造标量输入。

需要让 PIR `OpResult` 支持以下行为：

- `x != y`、`x.not_equal(y)` 和 `x.__ne__(y)` 生成等价的逐元素布尔结果。
- `x < y`、`x <= y`、`x > y`、`x >= y` 与对应的命名方法和 dunder 调用行为一致。
- 整数 `OpResult` 支持 `x & y`、`x | y`、`x ^ y` 和 `~x`，并与对应的位运算方法结果一致。
- `x.__pow__(2)` 与 `x ** 2` 一致，`x.__rpow__(2)` 与 `2 ** x` 一致。
- 上述表达式构造出的 PIR 程序可在 CPU executor 中运行，结果与对应的 NumPy 或动态图计算一致。

`x == y` / `x.__eq__(y)` 不属于本任务。它们的现有对象比较语义被 PIR 内部逻辑依赖，必须保持不变。

## 验收说明

- 不等和四种大小比较均返回可被 executor 获取的逐元素布尔结果，而不是 Python `bool`。
- `int32` 输入的四种位运算结果正确。
- `float32` 输入的标量正向和反向幂结果正确。
- 既有的整除、取模、矩阵乘法、属性访问和其他数学方法保持可用。
- 不允许通过删除测试、弱化断言、切回旧静态图、绕过 PIR executor 或改写 `__eq__` 语义来通过任务。

## 技术要求

- 熟悉 Python rich comparison 与位运算协议。
- 了解 Paddle PIR 静态图和 `OpResult` 的公开行为。
- 修复应局限于必要的 Python API 行为，不修改 C++ PIR backward、算子数值语义或底层 kernel。

## Acceptance Criteria

- The PIR `OpResult` comparison, bitwise, and scalar-power behavior described above should be implemented.
- Existing valid `OpResult` methods, attributes, and equality semantics should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, leaving PIR mode, bypassing execution, or redefining equality.

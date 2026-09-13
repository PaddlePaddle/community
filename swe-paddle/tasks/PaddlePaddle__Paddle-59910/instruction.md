# Make `paddle.distribution.Normal` work in PIR static programs

## 详细描述

在 PIR static graph 中使用 `paddle.distribution.Normal` 时，部分原本可在 dynamic graph 和旧 static graph 中使用的 Python / NumPy 参数形式无法正常完成参数转换。典型场景包括标量、list、tuple 或 NumPy array 形式的 `loc` / `scale`，以及这些参数之间存在 broadcast 的情况。

修复后，这些既有参数形式应能在 PIR static graph 中被正确转换，并保持与其他执行模式一致的 shape、dtype 和数值语义。

## 验收说明

- PIR static graph 下应支持既有 Python / NumPy 参数形式完成 `Normal` 参数转换。
- 存在 broadcast 的参数组合应保持正确 shape 和数值内容。
- 非 PIR 路径中的既有有效行为必须保持不变。

## 技术要求

- 理解 Paddle distribution 参数转换与 broadcast 语义。
- 理解 PIR static graph 与旧 static graph 的 Python API 差异。
- 能够维护不同执行模式之间的兼容行为。

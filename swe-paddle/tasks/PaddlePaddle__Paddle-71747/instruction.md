# 需要支持通用 iterable 的 reduce-like 操作

## 详细描述

在 SOT 执行路径中，将 generator 等通用 iterable 传给 `sum`、`max`、`min` 或 `map` 时，执行会触发 breakgraph，无法保持完整图运行。相同操作对部分具体容器已经有效，但依赖 Python iterator protocol 的程序仍无法在 dynamic-to-static 场景中获得一致的结果和消费顺序。

## 验收说明

- SOT 应支持对 generator 等通用 iterable 执行 `sum`、`max` 和 `min`，结果应与 eager execution 一致且不触发 breakgraph。
- `map` 和其他基于迭代消费的 reduce-like 调用应正确处理 generator，并保持元素顺序和返回值语义。
- 已有 generator、`reduce`、list、tuple 和其他容器行为应保持兼容。

## 技术要求

- 熟悉 Python iterator/generator protocol 与 builtin 函数语义
- 熟悉 Paddle SOT / dynamic-to-static 执行机制
- 熟悉 Variable dispatch、iterator abstraction 与 breakgraph 调试

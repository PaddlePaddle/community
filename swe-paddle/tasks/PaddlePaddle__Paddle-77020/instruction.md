# 完善 SOT 对 paddle.Size 的完整语义支持

## 详细描述

在 SOT 执行路径中使用 `paddle.Size` 时，构造、索引或切片、元素数量计算、加法、乘法、比较、容器方法和 symbolic shape 传播等操作不能稳定保持 eager execution 的结果及类型语义，部分场景还会触发 breakgraph。这会使依赖 shape 信息的 dynamic-to-static 程序无法作为完整图执行。

## 验收说明

- SOT 应支持 `paddle.Size` 的构造、索引与切片、`numel`、拼接、重复、比较及常用容器方法，并与 eager execution 的结果一致。
- 对 eager execution 应返回 `paddle.Size` 的操作，SOT 也应保持相同结果类型；symbolic shape 场景应能在不触发 breakgraph 的情况下执行。
- 已有 list 和 tuple 等容器的有效翻译行为应保持不变。

## 技术要求

- Python 容器与运算符语义
- Paddle SOT / dynamic-to-static 执行机制
- 变量分发、symbolic shape 与单元测试调试

## 参考资料

- https://github.com/PaddlePaddle/Paddle/pull/77020

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

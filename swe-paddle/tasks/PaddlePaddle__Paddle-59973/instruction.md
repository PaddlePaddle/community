# 新增 slice_scatter API

## 详细描述

为 Paddle 新增 `slice_scatter` 张量操作 API。它将 `value` 张量沿多个轴嵌入到 `x` 中,返回一个新张量而不是视图(类似 NumPy 的 slice-scatter 语义)。

要求:

- 新增 `paddle.slice_scatter(x, value, axes, starts, ends, strides, name=None)` 函数
- `axes`、`starts`、`ends`、`strides` 的长度必须相等
- 支持多轴同时嵌入,每个轴独立指定起始、结束与步长
- `value` 支持广播(broadcast)以匹配 slice 区域的形状
- `value` 与 `x` 的 `dtype` 应保持一致
- 在 `paddle` 顶层命名空间、`paddle.tensor` 命名空间及 tensor 方法中导出
- 支持的数据类型:bool, float16, float32, float64, uint8, int8, int16, int32, int64, bfloat16, complex64, complex128
- 动态图(PIR)与静态图两种模式均可用

## 验收说明

- `slice_scatter` 前向结果与 NumPy 对应操作一致(沿指定轴、起止与步长嵌入 value)
- 多轴嵌入与 value 广播行为正确
- 返回新张量,不修改原输入 `x`
- 静态图与动态图下均可用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

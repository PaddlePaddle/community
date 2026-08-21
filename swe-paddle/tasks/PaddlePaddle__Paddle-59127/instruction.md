# 新增 column_stack / row_stack / dstack / hstack / vstack API

## 详细描述

为 Paddle 新增 `column_stack`、`row_stack`、`dstack`、`hstack`、`vstack` 五个张量拼接 API,语义对齐 NumPy 同名函数。

要求:

- 新增 `paddle.hstack(x, name=None)`:水平拼接。所有输入先 `atleast_1d`;若为 1-D 输入沿 axis=0 拼接,否则沿 axis=1 拼接
- 新增 `paddle.vstack(x, name=None)`:垂直拼接。输入先 `atleast_2d`,沿 axis=0 拼接
- 新增 `paddle.dstack(x, name=None)`:深度拼接。输入先 `atleast_3d`,沿 axis=2 拼接
- 新增 `paddle.column_stack(x, name=None)`:将 1-D 输入先转成 2-D 列向量(`[N, 1]`)再沿 axis=1 拼接
- 新增 `paddle.row_stack(x, name=None)`:等价于 `vstack`
- 输入 `x` 为 Tensor 的 list/tuple,所有 Tensor 必须同 dtype;支持的数据类型:float16, float32, float64, int8, int32, int64, bfloat16
- 在 `paddle` 顶层命名空间、`paddle.tensor` 命名空间导出,并绑定为 tensor 方法

## 验收说明

- 五个 API 前向结果与 NumPy 对应函数一致(含 0-D/1-D/2-D 输入的补维行为)
- 拼接结果 shape 正确(沿各自 axis)
- 输入 dtype 不一致或形状不兼容时应按 NumPy 语义报错
- 静态图与动态图下均可用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

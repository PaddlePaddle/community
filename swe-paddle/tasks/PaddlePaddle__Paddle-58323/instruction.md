# 新增 atleast_1d / atleast_2d / atleast_3d API

## 详细描述

为 Paddle 新增 `atleast_1d`、`atleast_2d`、`atleast_3d` 三个张量操作 API。它们将标量或低维输入转换为至少具有指定维度的张量,高维输入保持不变。对标量输入(0-D)会先转换为 1-D 张量,不足目标维度时通过 reshape / unsqueeze 补维。

要求支持:

- `paddle.atleast_1d(*inputs, name=None)`:输出至少 1-D
  - 0-D 输入 reshape 为 `[1]`
  - 1-D 及以上输入原样返回
- `paddle.atleast_2d(*inputs, name=None)`:输出至少 2-D
  - 0-D 输入 reshape 为 `[1, 1]`
  - 1-D 输入 unsqueeze 到 `[1, N]`
  - 2-D 及以上输入原样返回
- `paddle.atleast_3d(*inputs, name=None)`:输出至少 3-D
  - 0-D、1-D、2-D 输入分别补维到 3-D
  - 3-D 及以上输入原样返回
- 支持可变参数:单输入返回单个 Tensor,多输入返回 Tensor 列表
- 输入可为 Tensor 或可被 `paddle.to_tensor` 转换的标量
- 在 `paddle` 顶层命名空间、`paddle.tensor` 命名空间及 tensor 方法中导出
- 支持的数据类型:float16, float32, float64, int16, int32, int64, int8, uint8, complex64, complex128, bfloat16, bool

## 验收说明

- `atleast_1d` / `atleast_2d` / `atleast_3d` 前向行为与 NumPy 对应 API(`np.atleast_1d/2d/3d`)一致
- 单输入返回 Tensor,多输入返回 Tensor 列表
- 高维输入保持不变,低维输入正确补维
- 静态图与动态图下均可用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

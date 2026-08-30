# 新增 matrix_exp API

## 详细描述

为 Paddle 新增 `matrix_exp` 线性代数 API,计算方阵的矩阵指数(matrix exponential)。矩阵指数定义为幂级数 `exp(A) = Σ A^k / k!`。

要求:

- 新增 `paddle.linalg.matrix_exp(x, name=None)` 函数
- 输入 `x` 为方阵(最后两维相等的任意 batch 形状),数据类型支持 float32 / float64(以及复数类型视底层支持)
- 通过 scaling(缩放矩阵以缩小范数)+ Padé 近似 + squaring(平方恢复)三步计算矩阵指数
- 在 `paddle.linalg` 命名空间及 tensor 方法中导出
- 静态图与动态图两种模式均可用

## 验收说明

- `matrix_exp` 前向结果与参考实现(如 SciPy `scipy.linalg.expm`、`np.linalg` 泰勒展开)一致
- 支持批量输入(batched square matrices)
- 结果与 `paddle.linalg.matrix_power` / 特征分解等间接验证方法一致
- 静态图与动态图下结果一致

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

# 新增 cholesky_inverse API

## 详细描述

为 Paddle 新增 `cholesky_inverse` 线性代数 API。给定对称正定矩阵的 Cholesky 分解因子,计算该矩阵的逆矩阵。

数学定义:

- `upper=False`(默认):`x` 为下三角 Cholesky 因子 `U`,满足 `A = UU^T`,返回 `inv = (UU^T)^{-1}`
- `upper=True`:`x` 为上三角 Cholesky 因子 `U`,满足 `A = U^TU`,返回 `inv = (U^TU)^{-1}`

要求:

- 新增 `paddle.linalg.cholesky_inverse(x, upper=False, name=None)` 函数
- 输入 `x` 必须为 2-D 方阵,否则应报错(维度不为 2 或行列不等)
- 支持 `upper` 参数选择下三角 / 上三角因子
- 数据类型支持 `float32`、`float64`
- 在 `paddle.linalg`、`paddle.tensor` 命名空间及 tensor 方法中导出

## 验收说明

- `paddle.linalg.cholesky_inverse` 前向结果与数学定义一致
- `upper=True/False` 两种模式结果一致(转置等价)
- 非 2-D 输入、非方阵输入应抛出 `ValueError`
- 与既有 `paddle.linalg.cholesky` 配合可正确还原原矩阵的逆

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

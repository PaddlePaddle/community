# 修复 `paddle.linalg.svd_lowrank` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.linalg.svd_lowrank(x)` 的输入 `x` 中存在大小为 `0` 的 dimension，即 `m` 和 `n` 中有一项为 0 时，当前实现在动态图模式下会因为参数校验和中间断言失败而报错。

典型表现包括：

- 参数校验阶段，当 `min(m, n) == 0` 时，`q >= 0 and q <= min(m, n)` 条件不满足，抛出 `ValueError`
- 中间计算阶段的 assert 检查在 0-size 维度上失败

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
x = paddle.to_tensor(np.random.random((1, 4, 0)).astype("float64"))
x.stop_gradient = False
out = paddle.linalg.svd_lowrank(x, q=4)
```

上述调用中 `x` 的 shape 为 `[1, 4, 0]`，`m=4, n=0`，`min(m, n)=0`。按照 API semantics，当输入 tensor 的某个维度为 0 时，SVD 分解应正常完成并返回正确 shape 的空 tensor。

当前实现层在进入参数校验和中间计算时，会失败而报错。

## 验收说明

- 当输入 tensor 的最后两个维度中有一个为 0 时，`paddle.linalg.svd_lowrank` 应正常完成，返回正确 shape 的空 tensor（U, S, V）
- 返回的 U、S、V 的 shape 应与非 0-size 输入时的 shape 推断一致
- 非 0-size tensor 输入下的 svd_lowrank 行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape、0-size Tensor 和动态图执行路径
- 了解 SVD 分解和多返回值语义
- 了解 Paddle 动态图和静态图执行路径的区别

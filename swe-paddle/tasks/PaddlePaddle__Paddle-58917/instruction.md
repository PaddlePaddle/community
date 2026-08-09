# 新增 tensor_split / hsplit / dsplit API

## 详细描述

为 Paddle 新增 `tensor_split`、`hsplit`、`dsplit` 三个张量切分 API,语义对齐 NumPy 同名函数。

要求:

- 新增 `paddle.tensor_split(x, num_or_indices, axis=0, name=None)`:
  - `num_or_indices` 为整数 `n` 时,沿 `axis` 切成 `n` 份;不能整除时前 `mod` 份大小为 `base+1`,其余为 `base`(不要求等分)
  - `num_or_indices` 为索引列表/元组时,在每个索引处切分,得到 `x[:i1]`、`x[i1:i2]`、…、`x[iN:]`;索引支持负值
  - `axis` 支持负值(等价于 `rank(x) + axis`)
  - 输入维度必须大于 0 且大于 `axis`,否则抛 `ValueError`
- 新增 `paddle.hsplit(x, num_or_indices, name=None)`:沿 `axis=1` 切分(即 `tensor_split(x, num_or_indices, axis=1)`)
- 新增 `paddle.dsplit(x, num_or_indices, name=None)`:沿 `axis=2` 切分(即 `tensor_split(x, num_or_indices, axis=2)`)
- 在 `paddle` 顶层命名空间、`paddle.tensor` 命名空间导出,并绑定为 tensor 方法
- 数据类型:bool, bfloat16, float16, float32, float64, uint8, int32, int64
- 既有 `vsplit` 行为保持不变(基于新实现重构)

## 验收说明

- `tensor_split` 整数切分(等分/非等分)与索引切分(含负索引)结果与 NumPy 一致
- `hsplit` / `dsplit` 结果分别与 `tensor_split` 沿 axis=1 / axis=2 一致
- 非法输入(维度不足、`num_or_indices <= 0`)应抛 `ValueError`
- 静态图与动态图下均可用

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

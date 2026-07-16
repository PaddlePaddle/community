# 完善 `paddle.put_along_axis` 和 `paddle.put_along_axis_` 对 0-size `indices` Tensor 的支持

## 详细描述

完善 out-of-place API `paddle.put_along_axis` 和 in-place API `paddle.put_along_axis_` 对 0-size `indices` Tensor 的支持。

本 issue 中，0-size Tensor 是指 `Tensor.shape` 中至少存在一个 size 为 `0` 的 dimension，因此其元素个数为 `0`，但仍具有 `shape`、`dtype`、`place` 等 Tensor metadata。

当 `indices` 为 0-size Tensor 时，不存在需要执行的 index update，API 应按 no-op semantics 正常完成。即使 `values` 的 shape 与 `indices` 不一致，也不应仅因 empty update path 中无需使用 `values` 而执行失败。与 empty update 无关的 dtype、rank、`axis` 等现有 validation 应保持不变。

## 验收说明

- 当 `indices` 为 0-size Tensor 时，接口应正常完成
- out-of-place API 应返回与 `arr` shape 和 data 一致的结果
- in-place API 应保持 `arr` 的 shape 和 data 不变
- 应覆盖 `values` 与 0-size `indices` shape 不一致但无需执行 update 的场景
- 非 0-size `indices` 的现有行为不得退化

## 技术要求

- 熟悉 Python 和 Paddle Tensor API
- 了解 Tensor shape 和 broadcast rules
- 了解 in-place 与 out-of-place API semantics
- 了解 Paddle 单元测试开发流程

## Acceptance Criteria

- The behavior described above should be supported correctly.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or broadly bypassing validation.

# 修复 `paddle.add_n` 0-size 与非 0-size 混合输入 shape 校验问题

## 详细描述

`paddle.add_n` 要求输入列表中的 Tensor 具有兼容的 shape。当第一个输入是 0-size Tensor 时，后续非 0-size Tensor 仍然必须与它满足相同的 shape 约束；如果二者不兼容，该 API 应按形状不一致拒绝输入，而不是继续计算或返回结果。

修复后，混合 0-size 与非 0-size 的不兼容输入应抛出 `ValueError`。同时，已经合法的全 0-size 输入仍应产生 0-size 输出，普通 shape 兼容的 Tensor 仍应按 `add_n` 语义进行逐元素求和。

## 验收说明

- 混合 0-size 与非 0-size 的不兼容输入应抛出 `ValueError`。
- 两个兼容的 0-size Tensor 仍应产生 shape 为 `[0]` 的输出。
- 两个普通兼容 Tensor 仍应返回逐元素求和结果。

## 技术要求

- 修复应聚焦于 `paddle.add_n` 可观察的 shape 校验行为。
- 保持合法 0-size 输入和普通兼容输入的既有行为不变。
- 不应通过删除或弱化测试、绕过验证、吞掉异常或宽泛放松 shape 校验来规避问题。

## Acceptance Criteria

- The behavior described above should be fixed.
- Existing valid behavior should remain unchanged.
- Do not satisfy the task by deleting tests, weakening assertions, or bypassing validation broadly.

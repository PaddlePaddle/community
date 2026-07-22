# `fused_dropout_add` 的结果可能与等价 `dropout + add` 不一致

## 详细描述

`paddle.incubate.nn.functional.fused_dropout_add` 在部分有效输入和执行场景中，可能产生与等价 `dropout + add` 组合不一致的数值结果。该差异会影响依赖数值一致性的 training 和 inference 流程。

在 optimized implementation 中已知的 precision issue 得到彻底解决前，应暂时避开受影响的 fused execution path，并保证该 API 保持等价 `dropout + add` 的 functional semantics。

当调用采用上述 compatibility behavior 时，应通过 warning 告知用户当前存在已知的 precision limitation。同一 module lifecycle 内不应重复发出相同 warning。

## 验收标准

- training 和 inference 场景均应保持等价 `dropout + add` 的 functional semantics
- `p`、`training` 和 `mode` 参数应继续生效
- 包括 `p == 0` 在内的现有有效参数组合应继续受支持，且结果不得退化
- 应向用户发出说明 temporary precision limitation 的 warning
- 同一 loaded module 中，相同 warning 最多发出一次

## 技术要求

- 熟悉 Python
- 了解 Paddle functional API 和 dropout semantics
- 了解 fused operator dispatch 与 fallback behavior
- 了解 warning lifecycle 和 Paddle 单元测试开发流程

## Acceptance Criteria

- The API produces the same observable result as the equivalent dropout-then-add composition.
- The fallback warning is emitted once per loaded module.
- Existing valid argument combinations remain supported.
- Do not satisfy the task by weakening tests or bypassing dropout semantics.

# PaddlePaddle__Paddle-77078

This directory converts Paddle PR #77078 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [77078](https://github.com/PaddlePaddle/Paddle/pull/77078) |
| PR title | [API Compatibility] Improve Cpp sink mechanism and sink paddle.inverse to cpp -part |
| Base commit | `f2de7486a07cbdbb6586771b5943df4bccc6d35c` |
| Merged at | `2026-01-30` (merge commit `78499bd`) |
| Related issue | [#76301](https://github.com/PaddlePaddle/Paddle/issues/76301) API 兼容性增强（启航计划） |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

将 `paddle.inverse` 下沉到 C++，补齐参数别名 `input` 与新的 `out` 参数（对齐 PyTorch），
并将 Cpp sink 的代码生成机制重构为支持任意模块路径，使 `paddle.inverse`、
`paddle.Tensor.inverse`、`paddle.linalg.inv` 三条路径在下沉后行为一致。

## Why This Sample

- **真实闭环**：下沉后触发真实 CE-Framework 测试失败（`paddle.linalg.inv` undefined），作者据此定位并修复。
- **代码生成 + API 兼容组合**：既要理解构建期 eager 代码生成器，又要处理多路径 API 语义对齐（`input` 别名、`out` 参数），是 benchmark 中稀缺的类型。
- **非平凡**：需要把只硬编码支持三个前缀的分类逻辑重构为支持任意模块路径的统一映射，而非简单加一条 YAML 配置。
- **边界清晰**：目标行为集中在三条路径的一致性与新参数，不改变 inverse 数值语义。

## Files

- `proposal.md`: approved proposal (maintainer triage context).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR (4 code files).
- `tests/test.patch`: test patch exposing the target behavior (`test_inverse_op.py`).
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit, build path, and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with `tests/test.patch` applied on `base_commit`, the new
compatibility cases (`input=` alias, `out=` parameter, `paddle.linalg.inv`
consistency) should fail/error. After also applying `solution/code.patch` and
rebuilding, the target tests should pass. Note this patch touches build-time code
generation, so a rebuild is required for the fix to take effect.

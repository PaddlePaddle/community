# PaddlePaddle__Paddle-76873

This directory converts Paddle PR #76873 and follow-up PR #77103 into one SWE-Paddle community task candidate (Hackathon 9th Sprint No.4: activation inplace support).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [76873](https://github.com/PaddlePaddle/Paddle/pull/76873) |
| Follow-up PR | [77103](https://github.com/PaddlePaddle/Paddle/pull/77103) |
| PR titles | `【Hackathon 9th Sprint No.4: Partial】... -part` / `【Hackathon 9th Sprint No.4】...` |
| Base commit | `471930236df5ba4e3bc34e1af6b8b9118e55a2d2` |
| Gold endpoint | `231207ce894f7f13e5c68e24cfa251ad41d10532` (after #77103) |
| Merged at | `2025-12-22` (#76873), `2026-01-13` (#77103) |
| Proposal | community `#1466` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

为 `CELU`、`RReLU`、`Swish`、`Mish`、`HardSigmoid`、`SELU` 等激活函数 API 补齐 inplace 能力，
使动态图与相关静态 / 符号形状路径行为一致。Gold 为两 PR 相对 base 的净效果。

## Why This Sample

- **真实 Hackathon 闭环**：先 partial 合入 `CELU`，再补齐其余激活函数。
- **API + 算子/符号形状组合**：不只改 Python 层，还需算子配置与 PIR 符号形状路径对齐。
- **双 PR 合一**：合并后样本对应完整 Sprint No.4 目标，而不是停留在中间态。
- **边界清晰**：只覆盖上述激活函数的 inplace；不要求一次覆盖全部 `paddle.nn.*`。

## Files

- `proposal.md`: approved proposal (do not modify in this package PR).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch net of #76873 + #77103 (production files only).
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit, build path, and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with `tests/test.patch` applied on `base_commit`, inplace /
compatibility cases for the listed activations should fail/error, while existing
non-inplace P2P cases can still pass. After also applying `solution/code.patch`
and rebuilding, all target cases should pass.

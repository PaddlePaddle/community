# PaddlePaddle__Paddle-78440

This directory converts Paddle PR #78440 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78440](https://github.com/PaddlePaddle/Paddle/pull/78440) |
| PR title | `[Fix] Fix paddle.cdist 0-size tensor handling: correct batch shape and stop_gradient propagation` |
| Base commit | `399e85b7d6c76c49af8301f718690c8d24548554` |
| Gold commit | `6d9f1d9f79d60d6b18760c03744ff331fa304737` |
| Merged at | `2026-03-24` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 `paddle.cdist` 处理零尺寸 Tensor 时的 batch shape 计算和 `stop_gradient` 传播问题。

## Why This Is A Good Candidate

- 来自 Paddle 主仓已合入的真实 bug-fix。
- 问题集中在 `paddle.cdist` 的零尺寸特殊分支，边界清楚。
- 同时覆盖高维 batch 广播和 Eager autograd 状态传播。
- 不依赖 GPU、网络服务或外部模型。
- Python-only patch，验证成本低，F2P/P2P 行为稳定。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: minimal regression tests exposing the target behavior.
- `tests/test.sh`: target verification command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should leave the P2P case passing while both F2P cases fail; applying both `tests/test.patch` and `solution/code.patch` should make all three target tests pass.

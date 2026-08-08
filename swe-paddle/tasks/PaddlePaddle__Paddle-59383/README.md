# PaddlePaddle__Paddle-59383

This directory converts Paddle PR #59383 and follow-up fix PR #60835 into one SWE-Paddle community task candidate (Hackathon No.4: add `masked_scatter` API).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [59383](https://github.com/PaddlePaddle/Paddle/pull/59383) |
| Follow-up PR | [60835](https://github.com/PaddlePaddle/Paddle/pull/60835) (small fix) |
| PR titles | `【Hackathon No.4】为 Paddle 新增 masked_scatter API -part` / `fix masked_scatter` |
| Base commit | `a8d5117371e8b9d16ff28011329bc04104eaf50a` |
| Gold endpoint | `a92999d0788ab7d4241a3daf9cadcb67566ef541` (after #60835) |
| Merged at | `2023-12-13` (#59383), `2024-01-17` (#60835) |
| Proposal | community `#1475` |
| Task type | `feature_enhancement` |
| Resource | CPU (pure Python) |

## Summary

为 Paddle 新增 `masked_scatter` / `masked_scatter_`，并纳入后续小修，使 mask 类型与静态图相关边界行为可用、可测。
Gold 为两 PR 相对 base 的净效果。

## Why This Sample

- **真实 Hackathon 闭环**：先合入新 API，再针对合入后问题做小修。
- **API 新增 + 边界修正**：不只暴露接口，还要保证动态图 / 静态图基本路径与错误处理正确。
- **双 PR 合一**：#60835 变更面很小，与主 PR 同属一个 API 目标。
- **验证成本低**：纯 Python，通常无需 C++ / codegen 重建。

## Files

- `proposal.md`: approved proposal (do not modify in this package PR).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch net of #59383 + #60835 (production files only).
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with `tests/test.patch` applied on `base_commit`,
`masked_scatter` F2P cases should fail/error. After also applying
`solution/code.patch`, target cases should pass. Existing unrelated inplace
cases remain P2P candidates.

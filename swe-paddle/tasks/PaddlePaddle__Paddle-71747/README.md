# PaddlePaddle__Paddle-71747

This directory converts Paddle PR #71747 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [71747](https://github.com/PaddlePaddle/Paddle/pull/71747) |
| PR title | `[SOT] Support reduce-like operation for all iterables` |
| Base commit | `243bad9d6a79d843b4db400c9f935a6d19ed48d0` |
| Merged at | `2025-03-18T16:55:04Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

完善 SOT 对 generator 等通用 iterable 的 reduce-like 操作支持，使 `sum`、`max`、`min` 和 `map` 等调用能够保持完整图执行和 Python 语义。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产代码跨六个 Python 文件并涉及 variable dispatch、iterator abstraction 和容器行为。
- 原 PR 自带 generator 行为测试，覆盖 `sum`、`max`、`min`、`reduce` 和 `map`。
- 生产代码 diff 为 165 行，具有明确的跨模块工程复杂度。
- 验证只需要 CPU，不依赖 Torch、GPU、CUDA、网络或外部数据集。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: original test diff from the merged PR exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: in the standard SOT strict-mode environment, applying `tests/test.patch` to `base_commit` should fail on unsupported generator reductions; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

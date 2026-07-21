# PaddlePaddle__Paddle-59909

This directory converts Paddle PR #59909 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [59909](https://github.com/PaddlePaddle/Paddle/pull/59909) |
| PR title | `[BugFix]Fix bug in parallel-mode select because of pure sharding` |
| Base commit | `6279a6784b35d96b910053b55ff6f763153e454a` |
| Merged at | `2023-12-12` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 hybrid parallel topology 在 sharding 与 data parallel 同时启用时错误选择 `DATA_PARALLEL` 的问题。

## Why This Is A Good SWE-Paddle Candidate

- It covers hybrid parallel topology and runtime parallel mode selection.
- It requires correct handling of sharding combined with data parallel.
- It preserves existing data parallel, pure sharding, tensor parallel, and pipeline parallel behavior.
- The target behavior can be verified deterministically on CPU without distributed communication.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target sharding-plus-data-parallel behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

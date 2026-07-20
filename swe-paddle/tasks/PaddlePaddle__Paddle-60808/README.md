# PaddlePaddle__Paddle-60808

This directory converts Paddle PR #60808 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | [60780](https://github.com/PaddlePaddle/Paddle/issues/60780) |
| PR | [60808](https://github.com/PaddlePaddle/Paddle/pull/60808) |
| PR title | `[BugFix]Fix broadcast_to bug while shape containing Zero-Dim` |
| Base commit | `161551046fd4c2b8a4ce19eb50fd6f5f0eeb5645` |
| Gold commit | `e2a324cb86120d2e82d9d9dbd9400d60e3a4bc8e` |
| Merged at | `2024-01-16` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 `paddle.broadcast_to` 在 static graph 或 dynamic-to-static 场景中处理包含 0-D Tensor dimension 的 `shape` list 时失败的问题。

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a real merged BugFix PR linked to issue #60780.
- The production change is limited to one Python file.
- The Base failure matches the reported observable behavior at graph construction time.
- The verifier executes the operation and checks output shape and broadcasted values.
- Existing integer-list and 1-D shape Tensor behavior is retained as P2P coverage.
- The task runs on CPU without rebuilding Paddle.

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

Expected behavior:

- Applying `tests/test.patch` to `base_commit` keeps the integer-list and 1-D shape Tensor P2P tests green.
- The 0-D Tensor shape-list test fails on `base_commit`.
- Applying both `tests/test.patch` and `solution/code.patch` makes all target tests pass.
- The patched production file must match the Git blob from `gold_commit`.

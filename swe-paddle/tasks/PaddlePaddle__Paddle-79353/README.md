# PaddlePaddle__Paddle-79353

This directory converts Paddle PR #79353 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79353](https://github.com/PaddlePaddle/Paddle/pull/79353) |
| PR title | `[Bug Fix] Fix p2p local_var bug` |
| Base commit | `406c7afec699c23158e7ff62a0f1afb306e72afe` |
| Gold commit | `15c873afa5dd01faeddbd39f6d985a69926c384e` |
| Merged at | `2026-06-23` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

完善 Pipeline P2P overlap 模式在无需执行通信时的返回行为。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target overlap paths while the non-overlap regression test passes; applying both patches should pass all target tests.

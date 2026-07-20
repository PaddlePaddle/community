# PaddlePaddle__Paddle-78922

This directory converts Paddle PR #78922 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78922](https://github.com/PaddlePaddle/Paddle/pull/78922) |
| PR title | `[FlexCheckPoint] fix memory leaking of a recursive function` |
| Base commit | `c55db2546c87e19fc78384c3497383298f3e2375` |
| Gold commit | `701baade42f4f0fdb8b1f43aff34f1cbc4db2061` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 FlexCheckpoint `flatten_state_dict` 调用结束后残留额外 Tensor reference 的问题，同时保持现有 flatten result、key mapping 和 unflatten behavior 不变。

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

Applying `tests/test.patch` to the base commit should fail the lifetime test while preserving the round-trip test. Applying `solution/code.patch` should make both tests pass.

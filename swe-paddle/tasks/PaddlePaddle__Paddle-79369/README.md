# PaddlePaddle__Paddle-79369

This directory converts Paddle PR #79369 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79369](https://github.com/PaddlePaddle/Paddle/pull/79369) |
| PR title | `【BugFix】Fix bug of check_memory_usage` |
| Base commit | `199073cd2021dd05efd8b0fe79797b838f68df41` |
| Gold commit | `adcaaa8f06fd435bd99452f2205f6ef49a53d19f` |
| Merged at | `2026-06-25` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

完善 Fleet `check_memory_usage` 在 CPU 运行环境中的内存日志行为。

## Why This Is A Good Candidate

- It comes from a merged Paddle bug-fix PR.
- The production change is isolated to one Python utility function.
- The target behavior is deterministic and covered with mocked device and system interfaces.
- It requires no GPU, distributed launch, external service, or C++ rebuild.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: regression tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: after applying `tests/test.patch` to the base commit, the P2P test passes and the F2P test fails. After also applying `solution/code.patch`, both tests pass.

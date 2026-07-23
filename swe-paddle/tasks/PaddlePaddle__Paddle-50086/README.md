# PaddlePaddle__Paddle-50086

This directory converts Paddle PR #50086 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [50086](https://github.com/PaddlePaddle/Paddle/pull/50086) |
| PR title | `[BugFix][ConditionalBlock] fix judgement about scope validation` |
| Base commit | `fe811625db37300f74064a52e80c130d7ae347ed` |
| Merged at | `2023-02-09` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 `ConditionalBlock` 重复执行时复用已经失效的 child scope，导致 new executor 使用无效 execution scope 的问题。

## Why This Is A Good SWE-Paddle Candidate

- It covers C++ control-flow runtime state and scope lifecycle.
- It requires reasoning about cached pointers, parent-child scope ownership, and repeated execution.
- It preserves first-run and legacy-executor behavior while changing the new-executor lifecycle contract.
- The verifier compiles the checked-out scope-selection logic in a lightweight native C++17 harness.
- No full Paddle build, wheel installation, GPU, or distributed runtime is required.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve first-run and legacy-executor behavior while failing repeated-execution and stale-scope lifecycle tests; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

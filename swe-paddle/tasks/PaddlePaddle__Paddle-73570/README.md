# PaddlePaddle__Paddle-73570

This directory converts Paddle PR #73570 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73570](https://github.com/PaddlePaddle/Paddle/pull/73570) |
| PR title | `[0-size Tensor Job2 No.87] Add 0-size Tensor support for masked_fill` |
| Base commit | `3efb8dbb51547f0235a402135c54ed83c2f12d61` |
| Gold commit | `70574f3ff130128d7cfed5a7bc50f2842137cc98` |
| Merged at | `2025-07-01` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.masked_fill` and `paddle.diag` to correctly handle 0-size tensors in CPU/GPU/XPU kernels by adding early-return logic and fixing gradient shape handling.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to missing early-return logic and incorrect gradient shape handling.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (C++ kernel changes).
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | masked_fill/diag F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

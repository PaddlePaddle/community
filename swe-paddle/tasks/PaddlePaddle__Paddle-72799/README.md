# PaddlePaddle__Paddle-72799

This directory converts Paddle PR #72799 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [72799](https://github.com/PaddlePaddle/Paddle/pull/72799) |
| PR title | `[0-size Tensor No.41、125、296] Add 0-size Tensor support for cumsum` |
| Base commit | `5bdf2fb5f13fd689d197f8f43c263fb9e83e3c90` |
| Gold commit | `aa9fc46162ab0d86dfb25e83a315e5c92f3702f3` |
| Merged at | `2025-05-21` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.cumsum`, `paddle.logcumsumexp` to correctly handle 0-size tensors in CPU/GPU/XPU kernels by adding early-return logic when output has 0 elements.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to missing early-return logic in cumsum kernels.
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

| Revision state | Existing behavior (P2P) | cumsum/logcumsumexp F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

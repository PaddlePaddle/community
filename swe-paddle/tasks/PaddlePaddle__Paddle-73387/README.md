# PaddlePaddle__Paddle-73387

This directory converts Paddle PR #73387 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73387](https://github.com/PaddlePaddle/Paddle/pull/73387) |
| PR title | `[0-size Tensor Job2 No.58] Add 0-size Tensor support for gather_tree` |
| Base commit | `57d91535621b3b793c2bd1e6d5dcc2801ed893fd` |
| Gold commit | `71179f5ae909c4577479beb77321f87b9b0b00ae` |
| Merged at | `2025-06-19` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `gather_tree` operator to correctly handle 0-size tensors by adding early-return logic in CPU/GPU kernels and skipping shape equality checks in InferMeta when input is 0-size.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to shape mismatch checks and kernel execution on empty data.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (C++ kernel and InferMeta changes).
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | gather_tree F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

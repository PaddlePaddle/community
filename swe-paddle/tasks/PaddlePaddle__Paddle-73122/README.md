# PaddlePaddle__Paddle-73122

This directory converts Paddle PR #73122 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73122](https://github.com/PaddlePaddle/Paddle/pull/73122) |
| PR title | `[0-size Tensor No.112] Add 0-size Tensor support for multi_dot` |
| Base commit | `2624aee95b82873848e34fc3e5673a1ac42f84c4` |
| Gold commit | `29b711ba8db211ed31b3354562f62fb5ce568b40` |
| Merged at | `2025-06-11` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.linalg.multi_dot` to correctly handle 0-size tensors in CPU/GPU kernels by adding early-return logic when any input has 0 elements, with special handling for the case where the output tensor has non-zero size but inputs contain 0-size dimensions.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to missing early-return logic in the multi_dot kernel.
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

| Revision state | Existing behavior (P2P) | multi_dot F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

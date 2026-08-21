# PaddlePaddle__Paddle-73702

This directory converts Paddle PR #73702 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73702](https://github.com/PaddlePaddle/Paddle/pull/73702) |
| PR title | `[0-size Tensor Job2 No.16、85] Add 0-size Tensor support for paddle.gather_nd` |
| Base commit | `a311cbc2ea18cadd6cde71de661091c60fcd9ce5` |
| Gold commit | `8b69643bc063fb36e8be4d7df265a8b3082e32f5` |
| Merged at | `2025-07-04` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.gather_nd` to correctly handle 0-size tensors in CPU/GPU/XPU kernels by adding special handling when the last dimension of index is 0, using tile kernel to broadcast the input tensor.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size index tensors due to missing special handling in gather_nd kernels.
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

| Revision state | Existing behavior (P2P) | gather_nd F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

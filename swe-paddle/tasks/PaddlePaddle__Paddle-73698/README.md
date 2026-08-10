# PaddlePaddle__Paddle-73698

This directory converts Paddle PR #73698 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73698](https://github.com/PaddlePaddle/Paddle/pull/73698) |
| PR title | `[0-size Tensor No.160、162、164] Add 0-size Tensor support for conv1d_transpose` |
| Base commit | `3ad243cee3fc076300af25b9c806c47a09d7aa5c` |
| Gold commit | `cd827ae7ab4552dfb563b070c569d1bf919bba78` |
| Merged at | `2025-06-30` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.nn.functional.conv1d_transpose`, `conv2d_transpose`, `conv3d_transpose` to correctly handle 0-size tensors in CPU/GPU/XPU kernels by adding early-return logic when input has 0 elements, and fixing InferMeta to correctly compute output shapes for 0-size inputs.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to missing early-return logic in conv_transpose kernels and incorrect InferMeta shape computation.
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

| Revision state | Existing behavior (P2P) | conv_transpose F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

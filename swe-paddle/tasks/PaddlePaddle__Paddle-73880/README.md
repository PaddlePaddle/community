# PaddlePaddle__Paddle-73880

This directory converts Paddle PR #73880 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73880](https://github.com/PaddlePaddle/Paddle/pull/73880) |
| PR title | `[0-size Tensor Job2 No.66] Add 0-size Tensor support for paddle.nn.functional.softmax_with_cross_entropy` |
| Base commit | `e1842d4ce364b6e8334c39a8807b256682185d23` |
| Gold commit | `4b55ef8495c5bdb91e6e96892f31d9cee63ed681` |
| Merged at | `2025-07-16` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ operator kernel (CPU/GPU/XPU) |

## Summary

Fix `paddle.nn.functional.softmax_with_cross_entropy` to correctly handle 0-size tensors in CPU/GPU/XPU kernels.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ kernel level and covers multiple backends (CPU, GPU, XPU).
- The failure is deterministic: the base revision fails when processing 0-size tensors in cross entropy kernels.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch.
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | softmax_with_cross_entropy F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

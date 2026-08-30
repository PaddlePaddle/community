# PaddlePaddle__Paddle-73821

This directory converts Paddle PR #73821 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73821](https://github.com/PaddlePaddle/Paddle/pull/73821) |
| PR title | `[0-size Tensor No.205] Add 0-size Tensor support for pad` |
| Base commit | `4c0a9e966c763e900222ee8457060b845b7e1664` |
| Gold commit | `2489cc099daafe0d75907e9c1be5a9cd0dbfbdfa` |
| Merged at | `2025-07-09` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel + Python API |

## Summary

Fix `paddle.nn.functional.pad` and related pad kernels (CPU/GPU/XPU) to correctly handle 0-size tensors by adding early return with `pad_value` fill for forward and zero-fill for backward, and handling 0-size pad tensor input in the Python API.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior spans C++ operator kernels (CPU/GPU/XPU) and the Python API layer.
- The failure is deterministic: the base revision fails when processing 0-size tensors in pad operations.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (C++ kernel + Python API changes).
- `tests/test.patch`: tests exposing the target behavior (F2P: `TestPadOp_ZeroSize2`, `TestPad3dOp_ZeroSize_Circular`, `TestPad3dOp_ZeroSize_Replicate`; P2P: `TestPadOp`).
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | pad F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

# PaddlePaddle__Paddle-73850

This directory converts Paddle PR #73850 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73850](https://github.com/PaddlePaddle/Paddle/pull/73850) |
| PR title | `[0-size Tensor No.118] Add 0-size Tensor support for paddle.linalg.triangular_solve` |
| Base commit | `917f720a58b3ed5aeb8a1ac0022fdbd76f3b2b4b` |
| Gold commit | `0a23433eddfd286cbdb8746240eaf662cd027c69` |
| Merged at | `2025-07-08` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ phi kernel + Python test |

## Summary

Fix `paddle.linalg.triangular_solve` to correctly handle 0-size tensors in both forward and backward passes. The forward kernel adds an early return when `x.numel() == 0 || y.numel() == 0`, and the backward kernel fills gradients with 0 when `out.numel() == 0`.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the triangular_solve phi kernels (CPU/GPU) and grad kernel impl.
- The failure is deterministic: the base revision fails when processing 0-size tensors in triangular_solve.
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

| Revision state | Existing behavior (P2P) | triangular_solve F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + test patch + solution patch | PASS | PASS |

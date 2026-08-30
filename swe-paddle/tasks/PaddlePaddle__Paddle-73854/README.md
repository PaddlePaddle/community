# PaddlePaddle__Paddle-73854

This directory converts Paddle PR #73854 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73854](https://github.com/PaddlePaddle/Paddle/pull/73854) |
| PR title | `[0-size Tensor Job2 No.60] Add 0-size Tensor support for paddle.nn.functional.instance_norm` |
| Base commit | `6ad407d1aaf34c41e193361d22f10c39946b715f` |
| Gold commit | `81a61590467551723341382599b950b4e92a6e1e` |
| Merged at | `2025-07-22` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ phi kernel + Python test |

## Summary

Fix `paddle.nn.functional.instance_norm` to correctly handle 0-size tensors in both forward and backward passes. The forward kernel adds an early return when `x.numel() == 0`, and the backward kernel fills gradients with 0. InferMeta is modified to remove the 0-size check and correctly handle dimension inference when C=0.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the instance_norm phi kernels (CPU/GPU/XPU) and InferMeta functions.
- The failure is deterministic: the base revision fails when processing 0-size tensors in instance_norm.
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

| Revision state | Existing behavior (P2P) | instance_norm F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

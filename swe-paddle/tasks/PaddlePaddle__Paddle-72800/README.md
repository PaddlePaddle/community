# PaddlePaddle__Paddle-72800

This directory converts Paddle PR #72800 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [72800](https://github.com/PaddlePaddle/Paddle/pull/72800) |
| PR title | `[0-size Tensor No.39、40] Add 0-size Tensor support for cummin/cummax` |
| Base commit | `705356a392f8cbdf0571cd60f8c0462eab424a80` |
| Gold commit | `beaa40bddb70ca079979c4a54f053aaf64d549ce` |
| Merged at | `2025-05-27` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.cummin` and `paddle.cummax` to correctly handle 0-size tensors in CPU/GPU kernels by adding early-return logic when output has 0 elements.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to missing early-return logic in cummin/cummax kernels.
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

| Revision state | Existing behavior (P2P) | cummin/cummax F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

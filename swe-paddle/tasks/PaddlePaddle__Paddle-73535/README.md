# PaddlePaddle__Paddle-73535

This directory converts Paddle PR #73535 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73535](https://github.com/PaddlePaddle/Paddle/pull/73535) |
| PR title | `[Accuracy diff No.112] Fix accuracy diff for paddle.nn.functional.conv1d API` |
| Base commit | `9c1900ce422e3398bfccf95d3d33ba2cfa91faed` |
| Gold commit | `f8e6a83f6cb3725902082e7fbb011d7c1e8f6406` |
| Merged at | `2025-06-24` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python API (conv1d float16 on CPU) |

## Summary

Fix `paddle.nn.functional.conv1d` to correctly handle float16 weight/bias on CPU by converting them to float32 before computation and casting the result back to float16.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python API layer (`conv1d`) and does not require C++ kernel changes.
- The failure is deterministic: the base revision fails when `conv1d` receives float16 weight/bias on CPU because CPU does not support float16 conv operations.
- The task has clear regression coverage for existing float32/float64 conv1d behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (Python API changes in `conv.py`).
- `tests/test.patch`: tests exposing the target behavior (adds `TestFunctionalConv1D_CPU_FP16`).
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | conv1d F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

# PaddlePaddle__Paddle-73855

This directory converts Paddle PR #73855 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73855](https://github.com/PaddlePaddle/Paddle/pull/73855) |
| PR title | `[0-size Tensor Job2 No.56] Add 0-size Tensor support for paddle.nn.functional.dice_loss` |
| Base commit | `0a23433eddfd286cbdb8746240eaf662cd027c69` |
| Gold commit | `1d3518f4ab0bb6f188e222152529f0b31f6acee3` |
| Merged at | `2025-07-08` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python Tensor API |

## Summary

Fix `paddle.nn.functional.dice_loss` to correctly handle 0-size tensors by removing the assertion that rejects inputs with any dimension equal to 0.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python Tensor API and does not require rebuilding C++ kernels.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to an explicit assertion check.
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

| Revision state | Existing behavior (P2P) | dice_loss F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

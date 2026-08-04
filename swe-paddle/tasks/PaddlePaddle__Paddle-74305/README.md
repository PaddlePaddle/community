# PaddlePaddle__Paddle-74305

This directory converts Paddle PR #74305 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74305](https://github.com/PaddlePaddle/Paddle/pull/74305) |
| PR title | `[0-size Tensor No.354、355] Add 0-size Tensor support for unique` |
| Base commit | `2e4a7184e806f1780f7695be46952d651993ed4e` |
| Gold commit | `28db7b832e42f69f48c3f95d651ae57e961aa433` |
| Merged at | `2025-07-31` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python Tensor API |

## Summary

Fix `paddle.unique` to correctly handle 0-size tensors in dynamic mode.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python Tensor API and does not require rebuilding C++ kernels.
- The failure is deterministic: the base revision fails when processing 0-size tensors in dynamic mode.
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

| Revision state | Existing behavior (P2P) | unique F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |
PASS | PASS |

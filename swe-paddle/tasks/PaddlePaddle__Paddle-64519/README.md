# PaddlePaddle__Paddle-64519

This directory converts Paddle PR #64519 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [64519](https://github.com/PaddlePaddle/Paddle/pull/64519) |
| PR title | 【Hackathon 6th No.2】【Typing】为 Paddle 新增 cholesky_inverse API -part |
| Base commit | `2d746f9719ddd35e9e9f1330b019d996bdafbfac` |
| Merged at | `2024-06-11T02:33:18Z` |
| Hackathon | `6th` task `2` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `cholesky_inverse` API for Paddle, which computes the inverse of a symmetric positive definite matrix from its Cholesky factor, with `upper` option for lower/upper triangular factor.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

# PaddlePaddle__Paddle-59715

This directory converts Paddle PR #59715 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [59715](https://github.com/PaddlePaddle/Paddle/pull/59715) |
| PR title | 【Hackathon 5th No.36】 为 Paddle 新增 matrix_exp API -part |
| Base commit | `3edda65cca8d44a1bddea70ac6f04f2b95430e9c` |
| Merged at | `2023-12-21T13:18:18Z` |
| Hackathon | `5th` task `36` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `matrix_exp` API for Paddle, which computes the matrix exponential of a square matrix using the scaling-and-squaring method with Padé approximants (orders 3/5/7/9), similar to TensorFlow/Eigen's implementation.

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

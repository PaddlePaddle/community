# PaddlePaddle__Paddle-64320

This directory converts Paddle PR #64320 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [64320](https://github.com/PaddlePaddle/Paddle/pull/64320) |
| PR title | 【Hackathon 6th No.17】为 Paddle 新增 sparse.mask_as API -part |
| Base commit | `605f5e20305db0e4932a20d3e0e6cf7d7d9631d8` |
| Merged at | `2024-06-07T09:10:27Z` |
| Hackathon | `6th` task `17` |
| Task type | `feature_enhancement` |
| Resource | CPU + GPU |

## Summary

Add `sparse.mask_as` API for Paddle, which extracts values from a dense tensor at positions indicated by a sparse mask and outputs a sparse tensor.

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

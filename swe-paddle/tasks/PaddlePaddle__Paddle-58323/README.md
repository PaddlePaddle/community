# PaddlePaddle__Paddle-58323

This directory converts Paddle PR #58323 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [58323](https://github.com/PaddlePaddle/Paddle/pull/58323) |
| PR title | 【Hackathon 5th No.33】为 Paddle 新增 atleast_1d / atleast_2d / atleast_3d API -part |
| Base commit | `431a0d53bd1578385ebcd3021f08d88ed6f75c70` |
| Merged at | `2023-11-16T12:29:50Z` |
| Hackathon | `5th` task `33` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `atleast_1d`, `atleast_2d`, `atleast_3d` APIs for Paddle, which convert scalar or low-dimensional inputs to tensors with at least the requested number of dimensions, preserving higher-dimensional inputs unchanged.

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

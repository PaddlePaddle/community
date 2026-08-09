# PaddlePaddle__Paddle-59127

This directory converts Paddle PR #59127 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [59127](https://github.com/PaddlePaddle/Paddle/pull/59127) |
| PR title | 【Hackathon 5th No.31】为 Paddle 新增 column_stack / row_stack / dstack / hstack / vstack API -part |
| Base commit | `cb5ff84d214b86b2409b3aa83ef7cd4ccd06374b` |
| Merged at | `2023-12-12T07:36:50Z` |
| Hackathon | `5th` task `31` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `hstack`, `vstack`, `dstack`, `column_stack`, `row_stack` APIs for Paddle, which join a sequence of tensors along different axes, with NumPy-compatible semantics.

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

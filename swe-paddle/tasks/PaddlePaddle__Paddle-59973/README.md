# PaddlePaddle__Paddle-59973

This directory converts Paddle PR #59973 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [59973](https://github.com/PaddlePaddle/Paddle/pull/59973) |
| PR title | 【Hackathon 5th No.28】为 Paddle 新增 slice_scatter API -part |
| Base commit | `9765ba805b40db5b00c8003d24cf45013ebf2420` |
| Merged at | `2023-12-26T11:04:21Z` |
| Hackathon | `5th` task `28` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `slice_scatter` API for Paddle, which embeds a `value` tensor into `x` along multiple axes (like NumPy's `slice_scatter` semantics), returning a new tensor instead of a view.

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

# PaddlePaddle__Paddle-59847

This directory converts Paddle PR #59847 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [59847](https://github.com/PaddlePaddle/Paddle/pull/59847) |
| PR title | 【Hackathon 5th No.38】为 Paddle 新增 FractionalMaxPool2d / FractionalMaxPool3d API -kernel |
| Base commit | `600fc2f0e758d28c85c738c57ade718bef6daec5` |
| Merged at | `2024-01-12T08:55:21Z` |
| Hackathon | `5th` task `38` |
| Task type | `feature_implementation` |
| Resource | CPU + GPU |

## Summary

Add `nn.FractionalMaxPool2d` / `nn.FractionalMaxPool3d` APIs for Paddle, including forward/backward kernels (CPU + GPU), Python functional and layer wrappers, and `return_mask` support.

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

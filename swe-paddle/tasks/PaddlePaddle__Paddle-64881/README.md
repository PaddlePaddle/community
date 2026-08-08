# PaddlePaddle__Paddle-64881

This directory converts Paddle PR #64881 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [64881](https://github.com/PaddlePaddle/Paddle/pull/64881) |
| PR title | 【Hackathon 6th No.8】NO.8 为 Paddle 新增 FeatureAlphaDropout API |
| Base commit | `d972f9ab8bb3d2ea5d1757a860ae45774e53b6eb` |
| Merged at | `2024-06-28T04:04:39Z` |
| Hackathon | `6th` task `8` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `FeatureAlphaDropout` layer and `feature_alpha_dropout` functional API for Paddle. Feature alpha dropout randomly masks out entire channels (feature maps) while preserving the self-normalizing property, based on the same masked alpha dropout implementation as `alpha_dropout`.

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

# PaddlePaddle__Paddle-56470

This directory converts Paddle PR #56470 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | [55883](https://github.com/PaddlePaddle/Paddle/issues/55883) |
| PR | [56470](https://github.com/PaddlePaddle/Paddle/pull/56470) |
| PR title | ``[API Enhancement] No.6 support single `int` input in UpsamplingNearest2D and UpsamplingBilinear2D`` |
| Base commit | `3568a99c5f6ff0e5fd528d43bd283fde34fe078b` |
| Merged at | `2023-08-28` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Added support for a single integer `size` input to `UpsamplingNearest2D` and `UpsamplingBilinear2D`.

## Why This Is A Good SWE-Paddle Candidate

- It adds a user-visible API capability rather than fixing an internal-only failure.
- The production change is Python-only and limited to two closely related layer constructors.
- Independent behavior tests can distinguish Base and Solution without GPU, source build, or external data.
- Existing list, tuple, and `scale_factor` inputs provide clear P2P regression coverage.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

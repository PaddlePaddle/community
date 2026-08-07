# PaddlePaddle__Paddle-74439

This directory converts Paddle PR #74439 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74439](https://github.com/PaddlePaddle/Paddle/pull/74439) |
| PR title | `[API compatibility] add paddle.ravel` |
| Base commit | `cb81162732f15ae02e82b07f8462e04b093c2464` |
| Merged at | `2025-08-08` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add a public `paddle.ravel` API that returns a one-dimensional flattened result for tensors of any rank, including scalars and empty tensors.

## Why This Is A Good SWE-Paddle Candidate

- It comes from a merged Paddle API-compatibility pull request with a small, well-defined production scope.
- The target behavior is user-visible through both `paddle.ravel` and `paddle.tensor.ravel`.
- The existing `flatten` API provides a stable regression guard for behavior that must remain unchanged.
- The Python-only implementation can be verified deterministically with AST overlays and controlled doubles without a Paddle source build.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to the base commit should preserve the existing `flatten` regression test while failing the new `ravel` behavior tests; applying both patches should make all target tests pass.

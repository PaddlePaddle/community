# PaddlePaddle__Paddle-74421

This directory converts Paddle PR #74421 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | https://github.com/PaddlePaddle/Paddle/pull/74421 |
| PR title | `[API compatibility] add msort api` |
| Base commit | `0b7e62cbfe2368b51e554cc7c0deb3bd94cad8f7` |
| Merged at | `2025-08-08` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add a public `paddle.msort` API that returns the input tensor sorted in ascending order along axis 0 while preserving the existing `paddle.sort` behavior.

## Why This Is A Good SWE-Paddle Candidate

- It comes from a merged API-compatibility change with a small and clearly bounded production diff.
- The new behavior is directly observable through the public sorting contract and can be separated cleanly from existing `sort` behavior.
- The implementation is Python-only and can be verified deterministically with an AST overlay and controlled doubles.
- The task does not require a Paddle source build, GPU, distributed execution, or external data.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only Gold patch.
- `tests/test.patch`: deterministic tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to the base commit keeps the existing `sort` regression test passing while the `msort` target test fails; applying both `tests/test.patch` and `solution/code.patch` makes all target tests pass.

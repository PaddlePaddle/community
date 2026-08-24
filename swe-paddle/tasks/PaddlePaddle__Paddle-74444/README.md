# PaddlePaddle__Paddle-74444

This directory converts Paddle PR #74444 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#74444](https://github.com/PaddlePaddle/Paddle/pull/74444) |
| PR title | `[API compatibility] add paddle nn.functional.dropout1d api` |
| Base commit | `607dd38aead3118af96495d50b9829c78b2ecfab` |
| Merged at | `2025-08-08` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add the public `paddle.nn.functional.dropout1d` API for channel-wise dropout on 2D and 3D inputs while preserving the behavior of existing dropout helpers.

## Why This Is A Good SWE-Paddle Candidate

- The task comes from a merged Paddle API-compatibility PR with only two production Python files.
- The target behavior is externally observable through input-rank validation, channel-axis dropout semantics, and public namespace exposure.
- The fail-to-pass boundary is clear because `dropout1d` is absent at the base commit.
- Existing `dropout2d` behavior can be protected independently with a deterministic P2P test.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch derived from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to the base commit should fail on the new `dropout1d` behavior while the P2P case remains valid; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

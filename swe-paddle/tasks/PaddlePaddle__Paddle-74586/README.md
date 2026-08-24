# PaddlePaddle__Paddle-74586

This directory converts Paddle PR #74586 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74586](https://github.com/PaddlePaddle/Paddle/pull/74586) |
| PR title | `[API compatibility] add scatter_add api` |
| Base commit | `e5c11eb4ab20851a6ab76bd0a85c8650b20b0692` |
| Merged at | `2025-08-21` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add the public `paddle.scatter_add` API so indexed source values are accumulated into an input tensor along a selected dimension, while preserving existing tensor-manipulation behavior.

## Why This Is A Good SWE-Paddle Candidate

- The change comes from a merged Paddle API-compatibility PR with a compact Python production scope.
- The target behavior is externally observable through indexed additive updates and public namespace exposure.
- The task has a clear fail-to-pass boundary because the API is absent at the base commit.
- Existing manipulation behavior can be protected with a stable P2P regression test.

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

Expected behavior: applying `tests/test.patch` to the base commit should fail on the new `scatter_add` behavior while the P2P case remains valid; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

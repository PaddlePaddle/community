# PaddlePaddle__Paddle-74594

This directory converts Paddle PR #74594 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74594](https://github.com/PaddlePaddle/Paddle/pull/74594) |
| PR title | `[API compatibility] add broadcast_shapes api` |
| Base commit | `20e50fb447d8e34bafb337c180ca28d77dfb82ca` |
| Merged at | `2025-08-15` |
| Task type | `new_feature` / `api_compatibility` |
| Resource | CPU |

## Summary

Add the public `paddle.broadcast_shapes` API so callers can compute the common broadcasted shape of zero, one, or multiple input shapes while preserving the existing two-shape `broadcast_shape` behavior.

## Why This Is A Good SWE-Paddle Candidate

- It comes from a merged Paddle API-compatibility PR and represents a real user-facing compatibility gap.
- The expected behavior is directly observable through returned shapes and exceptions rather than implementation-specific source structure.
- The production change is small and Python-only, with a clear separation between public exports and the multi-shape wrapper behavior.
- Stable P2P and F2P tests can run on CPU without a Paddle native source build or GPU.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained task statement.
- `solution/code.patch`: production-only Gold patch.
- `tests/test.patch`: independent regression tests.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: reproduction environment notes.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with only `tests/test.patch` applied to the Base commit, the existing `broadcast_shape` P2P test passes while the new multi-shape API tests fail. After applying `solution/code.patch`, all target tests pass.

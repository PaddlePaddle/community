# PaddlePaddle__Paddle-79268

This directory converts Paddle PR #79268 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79268](https://github.com/PaddlePaddle/Paddle/pull/79268) |
| PR title | `[API Compatibility] Add alias paddle.utils.data.DistributedSampler for paddle.io.DistributedBatchSampler` |
| Base commit | `722421e3a49eadf5ea774639c3d8147aced333ce` |
| Merged at | `2026-06-08` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

A new public entry point, `paddle.utils.data.DistributedSampler`, has been added. Additionally, a configurable shuffle seed has been introduced for the existing `DistributedBatchSampler`, ensuring consistent data ordering when the same seed and epoch are used, while maintaining support for the original seed-free invocation method.

## Why This Is A Good SWE-Paddle Candidate

- It adds a public data-loading API and parameter capability rather than fixing an internal-only bug.
- The user-visible contract covers public export, constructor argument forwarding, and deterministic shuffle behavior.
- Existing `DistributedBatchSampler` behavior provides a direct P2P regression boundary.
- The behavior can be tested deterministically in a single CPU process without launching distributed workers.
- The production patch is Python-only and does not require a Paddle source build.

## Patch Boundary

`solution/code.patch` contains only the three production files changed by the Gold commit:

- `python/paddle/io/dataloader/batch_sampler.py`
- `python/paddle/utils/data/__init__.py`
- `python/paddle/utils/data/distributed.py`

The original PR changes under `test/legacy_test/` are intentionally excluded. Independent benchmark tests are supplied only through `tests/test.patch`.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve existing distributed batch sampling behavior but fail the public `DistributedSampler` and configurable-seed cases; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

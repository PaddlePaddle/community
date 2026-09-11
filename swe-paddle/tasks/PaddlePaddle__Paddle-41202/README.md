# PaddlePaddle__Paddle-41202

This directory converts Paddle PR #41202 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#41202](https://github.com/PaddlePaddle/Paddle/pull/41202) |
| PR title | `Add AutoTune to reader.py for DataLoader` |
| Base commit | `23d1b3e8ed8187bfb3bd926934dd6cc71e691e53` |
| Merged at | `2022-04-22T04:31:39Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Allow DataLoader to sample a small portion of a dataset and automatically select a suitable worker count, while preserving the existing behavior when automatic tuning is disabled or unavailable on the current platform.

## Why This Is A Good SWE-Paddle Candidate

- The problem reflects a common DataLoader configuration cost: a fixed worker count may underuse the CPU or add unnecessary process overhead.
- The change spans configuration, dataset sampling, worker-count search, ordinary batch samplers, and distributed batch samplers.
- The merged PR includes a dedicated upstream test file covering enabled, disabled, and distributed-sampler scenarios.
- The behavior can be verified with a small synthetic dataset on CPU without downloading data or loading model weights.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR.
- `tests/test.patch`: benchmark regression test adapted from the merged PR test coverage so Base can collect the P2P/F2P roles before Gold-only API lookup occurs.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should allow all three role candidates to be collected, keep the disabled-mode P2P valid, and make the two enabled auto-tune scenarios fail when they look up the Gold-only API. Applying both `tests/test.patch` and `solution/code.patch` should make all three tests pass.

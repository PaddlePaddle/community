# PaddlePaddle__Paddle-54435

This directory converts Paddle PR #54435 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#54435](https://github.com/PaddlePaddle/Paddle/pull/54435) |
| PR title | `[LAUNCH] enable sort ip in launch` |
| Base commit | `56fd25b87196b84523b3cf25cc1637d1ca1b0d75` |
| Merged at | `2023-06-08T09:21:24Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Allow distributed launch to optionally assign node order and rank by numeric IPv4 address across HTTP and ETCD rendezvous while preserving the existing behavior by default.

## Why This Is A Good SWE-Paddle Candidate

- The task addresses deterministic node-rank assignment in multi-node launch jobs.
- The production change connects configuration parsing with both HTTP and ETCD peer synchronization paths.
- The tests cover numeric IPv4 ordering, local-rank calculation, configuration exposure, and disabled-mode compatibility.
- Verification uses deterministic in-memory clients and requires no network service, dataset, model, or GPU.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR.
- `tests/test.patch`: behavior tests exposing the target launch behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve existing launch behavior while the IP-sorting scenarios fail; applying both `tests/test.patch` and `solution/code.patch` should make the complete target test file pass.

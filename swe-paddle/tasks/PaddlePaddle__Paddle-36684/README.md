# PaddlePaddle__Paddle-36684

This directory converts Paddle PR #36684 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [36684](https://github.com/PaddlePaddle/Paddle/pull/36684) |
| PR title | `fleet support elastic scale up/down` |
| Base commit | `9a9345fa4dc77be655811d8e484b99cb9ff5f356` |
| Merged at | `2021-11-11T06:27:42Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Enable Fleet elastic jobs to use a node-count range and keep ranks, hosts, and endpoints consistent across scale-out and scale-in transitions.

## Why This Is A Good SWE-Paddle Candidate

- It covers a real distributed-launch workflow with clear behavior before and after the change.
- The production change spans range parsing, readiness decisions, host membership, and endpoint updates rather than a single local condition.
- The candidate executes assertions and scenarios from the source PR tests without requiring a live etcd service or GPU.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: source-PR tests plus the controlled source adapter.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

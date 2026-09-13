# PaddlePaddle__Paddle-79633

This directory converts Paddle PR #79633 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79633](https://github.com/PaddlePaddle/Paddle/pull/79633) |
| PR title | `[Distributed Strategy] Fix KV server hangs under concurrent requests` |
| Base commit | `58354a509a8d60b2cb3cdf6ead63a6c845eefd23` |
| Merged at | `2026-08-10T12:30:53Z` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

Fix the distributed launch KV server so concurrent registration and incomplete requests do not block other nodes from completing startup synchronization.

## Why This Is A Good SWE-Paddle Candidate

- The failure is observable as distributed launch requests hanging during node registration and synchronization.
- The upstream PR includes focused tests for concurrent requests, stalled connections, and clean shutdown.
- The production change is limited to one Python file and can be verified on CPU using loopback networking.
- The task does not require a GPU, external service, dataset, or model checkpoint.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: exact upstream test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

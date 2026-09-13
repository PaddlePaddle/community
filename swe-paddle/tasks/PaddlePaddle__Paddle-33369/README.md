# PaddlePaddle__Paddle-33369

This directory converts Paddle PR #33369 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [33369](https://github.com/PaddlePaddle/Paddle/pull/33369) |
| PR title | `ELASTIC 1 : fault tolerance` |
| Base commit | `4b9430a1f9ac2650a6a58e061f005acf8fc12fb3` |
| Merged at | `2021-06-21T06:06:29Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add an optional elastic lifecycle for fleet collective jobs so worker failures and membership changes can trigger a controlled restart or regroup.

## Why This Is A Good SWE-Paddle Candidate

- It models a real distributed-control failure rather than a numerical or device-specific edge case.
- The 495-line production change coordinates launch routing, process status, membership changes, and exit propagation.
- Deterministic CPU tests can exercise the real checkout control flow with controlled worker and launcher states.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

# PaddlePaddle__Paddle-18687

This directory converts Paddle PR #18687 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#18687](https://github.com/PaddlePaddle/Paddle/pull/18687) |
| PR title | `add parameter server launch` |
| Base commit | `d07ad4c6059db28c5f384a25190385742d9ba718` |
| Merged at | `2019-07-22T14:11:50Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add a local parameter-server launcher that starts the requested server and worker processes with consistent roles, endpoints, identifiers, and training-script arguments.

## Why This Is A Good SWE-Paddle Candidate

- The change exercises command parsing, environment construction, role assignment, process orchestration, and failure propagation as one focused feature.
- Its 151-line Python production change is substantial while remaining isolated to one launcher file.
- Deterministic CPU tests can validate the process contract without starting training, loading model weights, or requiring external services.
- A P2P case protects the existing collective launcher's training-argument forwarding behavior.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: behavior tests exposing the target feature.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should keep the existing launcher regression case passing while the parameter-server cases fail; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

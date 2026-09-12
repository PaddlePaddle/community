# PaddlePaddle__Paddle-40111

This directory converts Paddle PR #40111 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#40111](https://github.com/PaddlePaddle/Paddle/pull/40111) |
| PR title | `add profiler statistic helper` |
| Base commit | `10325a82e1032c3397b6f6611f558eb18ede0b07` |
| Merged at | `2022-03-08T01:55:55Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add deterministic interval operations for profiler statistics, including duration summation, normalization, union, intersection, and subtraction.

## Why This Is A Good SWE-Paddle Candidate

- The 225-line Python production change contains substantial interval-processing logic with meaningful edge cases.
- The source PR includes a focused 137-line unit test file, preserved byte-for-byte in the task patch.
- Tests cover sorted and unsorted input, overlap, containment, adjacency, disjoint ranges, empty ranges, and zero-length ranges.
- Verification is deterministic and CPU-only, with no operator execution, graph mode, model loading, network, or external service.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: exact upstream tests plus checkout-loading support and separate P2P coverage.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should keep the existing profiler scheduler case passing while the interval operations fail; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

# PaddlePaddle__Paddle-78932

This directory converts Paddle PR #78932 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78932](https://github.com/PaddlePaddle/Paddle/pull/78932) |
| PR title | `[API Compatibility] Support vararg and add alias for paddle.io.TensorDataset` |
| Base commit | `7b7e53fd28956700e5ed1ce68eb2aaeb59829777` |
| Merged at | `2026-05-11T08:51:45Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Allow `TensorDataset` to accept one or more tensors directly and expose it through `paddle.utils.data`, while preserving the existing list/tuple calling convention.

## Why This Is A Good SWE-Paddle Candidate

- The task reflects a common data-loading compatibility issue with clear inputs and observable dataset behavior.
- The change must distinguish a single Tensor from a list/tuple and from multiple positional tensors without breaking existing calls.
- The source PR provides real tests for list-based construction, single- and multi-Tensor varargs, item structure, dataset length, and public alias availability.
- The tests run deterministically on CPU without workers, external datasets, network access, or distributed devices.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: exact upstream test changes exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve the existing list-based cases but fail on direct Tensor arguments and the missing public alias; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

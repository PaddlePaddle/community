# PaddlePaddle__Paddle-78048

This directory converts Paddle PR #78048 into a SWE-Paddle community task candidate.

## Source

| Field       | Value                                                        |
| ----------- | ------------------------------------------------------------ |
| Repo        | `PaddlePaddle/Paddle`                                        |
| PR          | [78048](https://github.com/PaddlePaddle/Paddle/pull/78048)   |
| PR title    | `[API Compatibility No.62、73、234] Add parameter alias support for dsplit、hsplit、vsplit - part` |
| Base commit | `3f270c40db7776481d69176ee09222b3437d92bb`                   |
| Merged at   | `2026-03-05T10:11:27+08:00`                                  |
| Task type   | `feature_enhancement`                                        |
| Resource    | CPU                                                          |

## Summary

Allow `paddle.hsplit`, `paddle.dsplit`, and `paddle.vsplit` to accept commonly used parameter aliases while preserving their existing positional and Paddle-native keyword forms.

## Why This Is A Good SWE-Paddle Candidate

- The task comes from a real API compatibility gap with clear trigger conditions and expected results.
- The change covers three related public APIs through shared parameter-handling behavior rather than isolated special cases.
- The source PR provides real Tensor-based tests for the original names, the new aliases, Tensor methods, and NumPy reference results.
- The tests run deterministically on CPU without external datasets, network access, or distributed devices.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the unsupported parameter aliases; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

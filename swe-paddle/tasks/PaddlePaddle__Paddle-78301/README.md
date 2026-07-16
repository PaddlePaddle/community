# PaddlePaddle__Paddle-78301

This directory converts Paddle PR #78301 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [78301](https://github.com/PaddlePaddle/Paddle/pull/78301) |
| PR title | `[API Compatibility] align paddle.nn.Layer.to/paddle.Tensor.to -part` |
| Base commit | `fd041ffe2d941d7219090cb12f6ffb10860dc851` |
| Squash commit | `d6e41b70154ba52525884afdc15c2a9d763a2cae` |
| Merged at | `2026-04-03T07:08:36Z` |
| Track | `C` (feature_or_api) |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Align the public conversion call forms of `paddle.nn.Layer.to` and `paddle.Tensor.to`. The task covers positional and keyword dtype/device/reference-Tensor arguments, blocking options, `copy`, no-argument calls, invalid-input errors, and recursive in-place layer conversion.

## Why This Task

- It comes from a merged API-compatibility fix with a clear user-visible contract.
- It requires consistent overload parsing across two public APIs rather than a signature-only change.
- It exercises success paths, argument conflicts, invalid input, layer identity, and recursive sublayer conversion.
- Its Python-only scope is suitable for CPU verification while remaining non-trivial.

Only the behavior merged in PR #78301 is in scope. Follow-up PRs referenced by `proposal.md` are intentionally excluded from the gold patch and acceptance criteria.

## Files

- `proposal.md`: approved candidate proposal and scope.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch for non-test source files.
- `tests/test.patch`: target behavior tests, including a same-dtype positional `copy=True` case that requires a distinct result.
- `tests/test.sh`: minimal F2P/P2P test command.
- `environment/README.md`: base revision, run order, and verification risks.

## Verification

From a Paddle source checkout at the base commit, apply `tests/test.patch` and run:

```bash
bash tests/test.sh
```

The target tests should fail before `solution/code.patch` is applied and pass afterward. The existing `TestLayerTo::test_main` node is included as a P2P regression guard.

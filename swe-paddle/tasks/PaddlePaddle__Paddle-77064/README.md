# PaddlePaddle__Paddle-77064

This directory converts Paddle PR #77064 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [77064](https://github.com/PaddlePaddle/Paddle/pull/77064) |
| PR title | `[API Compatibility] Sink paddle.allclose to cpp -part` |
| Base commit | `1a6a9ab02e12fd792d036dc78b94f46a1371e6fa` |
| Squash commit | `407e3b6931a282a78653d559e675a598153ae977` |
| Merged at | `2025-12-25T11:41:33Z` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

Improve `allclose` API compatibility while preserving its numerical semantics. The public function accepts both its established argument names and the `input`/`other` aliases, the Tensor method accepts `other`, and a compatibility entrypoint returns a Python `bool` rather than a Tensor.

## Why This Task

- It comes from a merged API-compatibility change with deterministic, user-visible behavior.
- It spans a generated public binding, function and Tensor-method aliases, static-graph validation, and a compatibility API.
- It requires preserving two distinct return contracts: a scalar boolean Tensor for the primary API and Python `bool` for the compatibility API.
- The target behavior and regressions are verifiable on CPU with fixed inputs.

## Files

- `proposal.md`: candidate proposal and maintainer review context.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch for six non-test source files.
- `tests/test.patch`: test changes from the merged fix in two test files.
- `tests/test.sh`: F2P tests plus existing allclose regression coverage.
- `environment/README.md`: base revision, source-build path, run order, and risks.

## Verification

From a Paddle source checkout at the base commit, apply `tests/test.patch`, then run the task script while keeping the Paddle checkout as the working directory:

```bash
TASK_DIR=/path/to/community/swe-paddle/tasks/PaddlePaddle__Paddle-77064
bash "$TASK_DIR/tests/test.sh"
```

The alias and compatibility cases should fail before `solution/code.patch` is applied and pass after applying it and rebuilding Paddle. The remaining allclose module provides P2P regression coverage for numerical, input-dtype, tolerance, NaN, dynamic-graph, and static-graph behavior; the merged change intentionally removes obsolete Python-wrapper attribute-type assertions.

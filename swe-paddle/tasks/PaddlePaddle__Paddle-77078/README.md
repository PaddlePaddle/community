# PaddlePaddle__Paddle-77078

This directory converts Paddle PR #77078 into a candidate SWE-Paddle community task.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [77078](https://github.com/PaddlePaddle/Paddle/pull/77078) |
| PR title | [API Compatibility] Improve the C++ sinking mechanism and partially sink `paddle.inverse` to C++ |
| Base commit | `f2de7486a07cbdbb6586771b5943df4bccc6d35c` |
| Merged on | `2026-01-30` (merge commit `78499bd`) |
| Related issue | [#76301](https://github.com/PaddlePaddle/Paddle/issues/76301) API Compatibility Enhancement (Sailing Program) |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

Sink `paddle.inverse` to C++, add support for the `input` parameter alias and the new `out` parameter to align with PyTorch, and refactor the C++ sinking code-generation mechanism to support arbitrary module paths. This ensures that `paddle.inverse`, `paddle.Tensor.inverse`, and `paddle.linalg.inv` behave consistently after the API is sunk to C++.

## Why This Sample

- **Real end-to-end failure and fix**: After the API was sunk to C++, an actual CE-Framework test failure occurred because `paddle.linalg.inv` was undefined. The PR author identified and fixed the issue based on this failure.
- **Combination of code generation and API compatibility**: The task requires understanding the build-time eager code generator and aligning API semantics across multiple entry points, including the `input` alias and `out` parameter. This combination is uncommon in the benchmark.
- **Non-trivial implementation**: The existing classification logic, which hard-coded support for only three prefixes, must be refactored into a unified mapping that supports arbitrary module paths. The task cannot be solved by simply adding one YAML configuration entry.
- **Well-defined scope**: The target behavior is limited to consistency across the three API paths and support for the new parameters. The numerical semantics of `inverse` remain unchanged.

## Files

- `proposal.md`: Approved proposal containing maintainer triage context.
- `instruction.md`: Self-contained problem statement for the coding agent.
- `solution/code.patch`: Gold patch from the merged PR, covering four source files.
- `tests/test.patch`: Test patch that exposes the target behavior in `test_inverse_op.py`.
- `tests/test.sh`: Minimal command for running the target tests.
- `environment/README.md`: Base commit, build instructions, and reproduction notes.

## Verification

```bash
bash tests/test.sh
````

Expected behavior: With `tests/test.patch` applied to `base_commit`, the new compatibility cases—the `input=` alias, the `out=` parameter, and consistency with `paddle.linalg.inv`—should fail or raise errors. After applying `solution/code.patch` as well and rebuilding the project, the target tests should pass.

Note that this patch modifies build-time code generation, so a rebuild is required for the fix to take effect.


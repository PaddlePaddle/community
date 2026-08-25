# PaddlePaddle__Paddle-58917

This directory converts Paddle PR #58917 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [58917](https://github.com/PaddlePaddle/Paddle/pull/58917) |
| PR title | 【Hackathon 5th No.32】为 Paddle 新增 tensor_split / hsplit / dsplit API -part |
| Base commit | `46e3dfeaa50ec97edeebb1acd5205f5cd702bf5c` |
| Merged at | `2023-12-13T03:42:02Z` |
| Hackathon | `5th` task `32` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `tensor_split`, `hsplit`, `dsplit` APIs for Paddle. `tensor_split` splits a tensor along a given axis into either a fixed number of sections or at given indices (sections need not be equal); `hsplit` / `dsplit` are axis-specific wrappers (axis 1 / axis 2).

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

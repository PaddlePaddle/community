# PaddlePaddle__Paddle-56135

This directory converts Paddle PR #56135 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [56135](https://github.com/PaddlePaddle/Paddle/pull/56135) |
| PR title | `[BugFix] fix bmm op bugs in static mode with dynamic shape` |
| Base commit | `4f2cf7fbcaca52bb9625dc6be944f552ea1d71d5` |
| Merged at | `2023-08-16` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 `paddle.bmm` 在 static graph 下处理包含 unknown dimension 的 dynamic shape 时，错误拒绝合法输入或无法正确推导 output shape 的问题。

## Why This Is A Good SWE-Paddle Candidate

- It covers both Python static-graph validation and C++ infermeta behavior.
- It requires consistent handling of unknown dimensions across two framework layers.
- It preserves known-shape execution and known-incompatible-shape validation.
- The C++ verifier compiles the exact checked-out `BmmInferMeta` function in a lightweight native harness.
- It avoids a full Paddle rebuild while still executing the changed C++ control flow.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the Python static-graph and C++ infermeta dynamic-shape behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

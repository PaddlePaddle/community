# PaddlePaddle__Paddle-53534

This directory converts Paddle PR #53534 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [53534](https://github.com/PaddlePaddle/Paddle/pull/53534) |
| PR title | `【BugFix】fix err of api to_tensor, which caused by numpy version update` |
| Base commit | `f74237cd73c35b8a63d7981a190a302d0ebcd03f` |
| Merged at | `2023-05-08` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 static graph 中 `to_tensor` 在 NumPy 1.24+ array-construction semantics 下无法处理包含 Tensor/Variable 的 nested sequence，并改进 unsupported input type 的错误反馈。

## Why This Is A Good SWE-Paddle Candidate

- It covers a real compatibility regression caused by a NumPy behavior change.
- It requires preserving static-graph Tensor/Variable handling while supporting nested sequences.
- It validates output values, dtype conversion, `stop_gradient`, stack/squeeze behavior, and unsupported-type errors.
- The verifier executes the checked-out `_to_tensor_static` implementation through Python AST with controlled Paddle and NumPy doubles.
- The upstream unit-test change is intentionally excluded from `solution/code.patch`; this task uses an independent regression test.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve existing numeric and Variable behavior while failing on NumPy 1.24-style nested-sequence conversion and unsupported mapping errors; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

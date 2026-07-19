# PaddlePaddle__Paddle-54625

This directory converts Paddle PR #54625 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [54625](https://github.com/PaddlePaddle/Paddle/pull/54625) |
| PR title | [BugFix] fix bug of release output in pp |
| Base commit | `974676bc6ec41e222083729af55f34e4b2f20f2e` |
| Merged at | `2023-06-16` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 pipeline parallel 在释放中间输出时错误清理未初始化或已发生 in-place 修改的 Tensor 数据的问题。

## Why This Is The Starter Example

This sample is suitable as a SWE-Paddle starter example because:

- It covers a real Tensor-lifecycle bug in pipeline parallel output release.
- It requires preserving normal output cleanup while protecting unsafe Tensor states.
- It has a clear single-file production scope and deterministic P2P/F2P behavior.
- It can be verified on CPU with an AST overlay and controlled doubles, without GPU or distributed communication dependencies.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

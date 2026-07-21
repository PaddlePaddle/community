# PaddlePaddle__Paddle-56705

This directory converts Paddle PR #56705 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [56705](https://github.com/PaddlePaddle/Paddle/pull/56705) |
| PR title | [BugFix]Fix memory leak in mplayers |
| Base commit | `23955fcfab3ecf5bfe4be9d3a4543cb0d9c7c377` |
| Merged at | `2023-08-29` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 dynamic graph model-parallel identity 和 all-reduce 操作重复调用时产生的 Python-side runtime type 累积与长期内存增长问题。

## Why This Is The Starter Example

This sample is suitable as a SWE-Paddle starter example because:

- It covers a real memory-lifecycle bug in distributed model-parallel helpers.
- It requires preserving the existing forward/backward communication behavior while fixing repeated runtime type creation.
- It has a clear single-file production scope and deterministic P2P/F2P behavior.
- It can be verified on CPU with an AST overlay and controlled doubles, without GPU or multi-process timing dependencies.

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

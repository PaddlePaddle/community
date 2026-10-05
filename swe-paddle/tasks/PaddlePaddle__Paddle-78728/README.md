# PaddlePaddle__Paddle-78728

This directory converts Paddle PR #78728 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78728](https://github.com/PaddlePaddle/Paddle/pull/78728) |
| PR title | `support to detect the recompute context` |
| Base commit | `52a2dfd7f3f7f402fe13161bd6d6bce5e9a727fa` |
| Merged at | `2026-04-22T07:52:37Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

为 Fleet recompute 增加可查询的执行上下文，使下游 Python 代码能够区分普通前向计算与 recompute 执行，同时保证异常恢复、线程隔离和原有训练语义。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产改动跨越 recompute 实现与公开导出两个 Python 文件。
- 原 PR 新增 281 行完整行为测试，覆盖上下文管理器、装饰器、异常清理、线程隔离、前后向一致性和训练步骤。
- production diff 为 111 行，测试补丁与生产补丁边界清楚。
- 验证只需要 CPU，不依赖 Torch、GPU、CUDA、网络或外部数据集。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR.
- `tests/test.patch`: exact original test diff from the merged PR.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Applying `tests/test.patch` to `base_commit` should fail because the recompute-context API is unavailable. Applying both patches should make all 14 original PR tests pass.

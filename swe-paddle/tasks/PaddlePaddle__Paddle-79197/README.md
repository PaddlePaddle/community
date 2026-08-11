# PaddlePaddle__Paddle-79197

This directory converts Paddle PR #79197 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79197](https://github.com/PaddlePaddle/Paddle/pull/79197) |
| PR title | `[API Compatibility] Support param optimizer for lr_scheduler` |
| Base commit | `06d8af53d39ef6622689bab27e1cd03a2ffab0f3` |
| Merged at | `2026-06-08T06:44:24Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

让常用 learning-rate scheduler 可以直接接收已有 optimizer，并自动与该 optimizer 建立关联，同时保持原有 `learning_rate` 调用方式不变。

## Why This Is A Good SWE-Paddle Candidate

- 问题来自用户迁移训练代码时常见的 scheduler 调用方式差异，触发条件和期望结果清楚。
- 修改覆盖多个 scheduler 的统一入口和参数处理逻辑，不能靠单点特判完成。
- 来源 PR 提供了位置参数、关键字参数、学习率变化和 optimizer 关联关系的真实测试。
- 测试可在 CPU 环境运行，不需要外部数据集、网络或分布式设备。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: exact upstream diff for `test/legacy_test/test_lr_scheduler.py`.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail when schedulers receive an optimizer; applying both `tests/test.patch` and `solution/code.patch` should pass the complete upstream test file.

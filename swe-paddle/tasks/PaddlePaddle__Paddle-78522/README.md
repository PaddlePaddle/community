# PaddlePaddle__Paddle-78522

This directory converts Paddle PR #78522 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78522](https://github.com/PaddlePaddle/Paddle/pull/78522) |
| PR title | `[Distributed] Replace \`os.system\` with \`os.kill\` in \`launch/main.py\`` |
| Base commit | `3493dbf5fdf1da8e59b8de87ec268ab386b9eefb` |
| Merged at | `2026-04-08T09:19:01Z` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复分布式 auto-tuner 清理设备占用进程时，对异常进程列表输入和进程退出竞态处理不稳健的问题。

## Why This Is A Good SWE-Paddle Candidate

- 真实来源于已合入的分布式启动改进，行为边界集中在任务间进程清理。
- upstream test 使用 mock 验证 PID 过滤、调用顺序和异常语义，无需 GPU、网络或真实子进程竞态。
- Base 缺少目标清理契约，upstream test 可稳定形成 F2P；已有 launch utility 测试用于保护 P2P。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: exact upstream test patch from the merged PR.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

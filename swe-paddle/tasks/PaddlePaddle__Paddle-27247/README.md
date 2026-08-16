# PaddlePaddle__Paddle-27247

This directory converts Paddle PR #27247 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [27247](https://github.com/PaddlePaddle/Paddle/pull/27247) |
| PR title | `move DataLoader._worker_loop to top level` |
| Base commit | `aae41c6fca67be6a090d4f83bdf6160737d15162` |
| Merged at | `2020-09-14T05:51:32Z` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复多进程 DataLoader 在 `spawn` 启动方式下因进程任务无法序列化而不能启动 worker 的问题。

## Why This Is A Good SWE-Paddle Candidate

- 来源 PR 给出了 `paddle.distributed.spawn` 配合多进程 DataLoader 时的真实报错和完整调用栈。
- 修改同时覆盖新版 DataLoader worker 和 legacy generator reader 两条路径，需要保持原有启动与队列协作行为。
- 原 PR 的两份测试改动被原样保留；新增的行为测试直接验证进程启动参数能否通过 Python multiprocessing 的 `spawn` 序列化阶段。
- 测试使用 checkout 中的真实控制流，可在 CPU 环境稳定运行，不启动训练任务，也不依赖 GPU、网络或外部数据集。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: upstream test updates plus the behavior test exposing the target failure.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

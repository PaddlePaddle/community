# PaddlePaddle__Paddle-59910

This directory converts Paddle PR #59910 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | [#58067](https://github.com/PaddlePaddle/Paddle/issues/58067) |
| PR | [#59910](https://github.com/PaddlePaddle/Paddle/pull/59910) |
| PR title | `【PIR API adaptor No.31】Migrate paddle.distribution.Normal into pir` |
| Base commit | `5b3bc5043bcbc919a4bb9e69f421d6001814b365` |
| Merged at | `2024-01-16T03:31:00Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

使 `paddle.distribution.Normal` 在 PIR static graph 下正确处理既有 Python / NumPy 参数及 broadcast，同时保护 legacy-mode 行为。

## Why This Is A Good SWE-Paddle Candidate

- Production change 仅涉及一个 Python 文件，不需要重新编译 Paddle。
- 原 PR 自带对应 distribution 单测修改，目标行为边界明确。
- task 使用独立 `test/swe_paddle` contract test，从 checkout 源码执行真实 Python 控制流，避免历史 build-tree import path 依赖。
- CPU 上即可稳定验证，不依赖 GPU、NCCL、网络、外部数据或随机时序。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: independent behavior test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should keep the P2P behavior passing while the PIR-targeted F2P behavior fails; applying both `tests/test.patch` and `solution/code.patch` should make all target tests pass.

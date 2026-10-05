# PaddlePaddle__Paddle-77020

This directory converts Paddle PR #77020 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [77020](https://github.com/PaddlePaddle/Paddle/pull/77020) |
| PR title | `[SOT] Add fully \`SizeVariable\` support` |
| Base commit | `c8f9b3d44f61dd80d2c06b40000d1c21f7dca427` |
| Merged at | `2025-12-23T17:05:27Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

完善 SOT 对 `paddle.Size` 构造、索引、算术、比较和 shape 传播等行为的支持，使其在不触发 breakgraph 的情况下与 eager execution 保持一致。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产代码跨四个 Python 文件，具有明确的模块协作边界。
- PR 自带完整行为测试，覆盖 `paddle.Size` 的构造、索引、运算、方法调用和 symbolic shape 场景。
- 验证只需要 CPU，不依赖 Torch、GPU、CUDA、网络或外部数据集。
- Base 上可复现目标测试失败，同时已有 list 运算保持通过；Gold patch 后新增测试全部通过。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: original test diff from the merged PR exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

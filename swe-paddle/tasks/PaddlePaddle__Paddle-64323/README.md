# PaddlePaddle__Paddle-64323

This directory converts Paddle PR #64323 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [64323](https://github.com/PaddlePaddle/Paddle/pull/64323) |
| PR title | `[SOT][dynamic shape] Support dynamic int input` |
| Base commit | `34577c4c9a14a808fd6abec145f796b30ec93c60` |
| Merged at | `2024-05-19T12:57:37Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

为 SOT 增加动态整数输入支持，使整数参数在达到动态化条件后能够作为符号值参与 Tensor 形状和算术计算，并复用已翻译的动态图缓存。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产改动横跨 SOT 缓存、函数图、变量跟踪、分派、符号变量、语句 IR 和环境配置等 11 个 Python 文件。
- 原 PR 自带完整行为测试，通过连续输入五个不同整数检查翻译次数和动态缓存复用。
- production diff 为 283 行，包含符号整数建模、运算分派、输入提升和缓存策略等工程改动。
- 验证只需要 CPU，不依赖 Torch、GPU、CUDA、网络或外部数据集。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: exact production patch from the merged PR.
- `tests/test.patch`: exact original test diff from the merged PR.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Applying `tests/test.patch` to `base_commit` should fail because the dynamic-shape guard and symbolic integer support are unavailable. Applying both patches should make the original PR test pass.

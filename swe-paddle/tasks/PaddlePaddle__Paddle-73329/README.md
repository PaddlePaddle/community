# PaddlePaddle__Paddle-73329

This directory converts Paddle PR #73329 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73329](https://github.com/PaddlePaddle/Paddle/pull/73329) |
| PR title | `[SOT] Support \`dataclass\` construction and basic access` |
| Base commit | `eb9af7cf69977981198f41ae79eebabd0e3e036f` |
| Merged at | `2025-06-15T09:50:03Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

完善 SOT 对 Python `dataclass` 构造及实例访问的支持，使字段默认值、default factory、属性读写和比较等行为在 strict mode 下保持 eager execution 语义。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产代码跨七个 Python 文件并涉及变量建模、分发、side effect 和函数图。
- 原 PR 自带 dataclass 行为测试，覆盖构造、默认字段、属性访问与更新以及实例比较。
- 生产代码 diff 超过 300 行，具有明确的跨模块工程复杂度。
- 验证只需要 CPU，不依赖 Torch、GPU、CUDA、网络或外部数据集。

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

Expected behavior: in the standard SOT strict-mode environment, applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

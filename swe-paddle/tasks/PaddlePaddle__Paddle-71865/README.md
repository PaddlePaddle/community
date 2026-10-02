# PaddlePaddle__Paddle-71865

This directory converts Paddle PR #71865 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [71865](https://github.com/PaddlePaddle/Paddle/pull/71865) |
| PR title | `[SOT] Support builtin function \`super\`` |
| Base commit | `8c6c18cbe059e452bb18209f83b9dbc6c3d7d436` |
| Merged at | `2025-03-28T12:44:35Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

完善 SOT 对 Python 内置 `super()` 的支持，使单继承、多继承、属性访问、显式参数和隐式参数等调用在不触发 breakgraph 的情况下保持 Python 语义。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实工程 PR，生产代码跨七个 Python 文件并涉及字节码执行、变量分发、方法绑定和 guard。
- 原 PR 自带完整行为测试，覆盖单继承、多继承、`super()` 作为输入、属性查找、缓存 guard 和自定义同名函数。
- 生产代码 diff 超过 100 行，不能通过单点条件判断或测试特判完成。
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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

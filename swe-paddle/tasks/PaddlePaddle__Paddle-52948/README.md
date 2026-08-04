# PaddlePaddle__Paddle-52948

This directory converts Paddle PR #52948 and follow-up PR #53572 into one SWE-Paddle community task candidate (Hackathon No.91: `register_hook` for static / dy2static).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [52948](https://github.com/PaddlePaddle/Paddle/pull/52948) |
| Follow-up PR | [53572](https://github.com/PaddlePaddle/Paddle/pull/53572) |
| PR titles | `【Hackathon No.91】` / `【Hackathon No.91】Following updates` |
| Base commit | `cf6cbc347970a1fd2c9d76e427880139789497af` |
| Gold endpoint | `f3f3d57a159caf3b77f93a4d86cb233e6a1c159a` (after #53572) |
| Merged at | `2023-04-27` (#52948), `2023-05-08` (#53572) |
| Task type | `feature_enhancement` |
| Resource | CPU (pure Python; era-matched Paddle / source checkout) |

## Summary

在静态图与动转静（`to_static`）场景下支持 `Tensor.register_hook`，使反向 hook
能正确触发，且梯度结果与动态图一致。完整 gold 覆盖首个合入实现及其后续规范化
接入（独立 Transformer 进入统一 AST 变换流水线、测试断言风格对齐）。

## Why This Sample

- **真实 Hackathon 闭环**：合入前相关单测显式断言 static / dy2static 下 hook 不可用。
- **框架级能力**：同时覆盖静态图 hook 运行时语义与动转静路径下的行为对齐。
- **双 PR 合一**：#52948 落地能力，#53572 按 review 意见完成 Transformer 规范化；合并后样本对应当前更合理的最终形态。
- **边界清晰**：目标集中在 hook 可运行且梯度一致；不要求内部函数 hook 与 `hook.remove`。

## Files

- `proposal.md`: approved proposal (maintainer triage context).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch net of #52948 + #53572 (production files only).
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit, apply order, and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with `tests/test.patch` applied on `base_commit`, the static /
dy2static `register_hook` cases should fail/error. After also applying
`solution/code.patch`, the target tests should pass. Existing dygraph
`register_hook` cases in the same module should remain pass-to-pass.

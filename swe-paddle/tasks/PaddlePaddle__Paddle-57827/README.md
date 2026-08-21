# PaddlePaddle__Paddle-57827

This directory converts Paddle PR #57827 into a SWE-Paddle community task candidate (PIR: `fused_elemwise_add_activation`).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [57827](https://github.com/PaddlePaddle/Paddle/pull/57827) |
| PR title | `【PIR】fused_elemwise_add_activation` |
| Base commit | `8b1a29ba9bafc16116f97422574e85d208540332` |
| Gold endpoint | `3ac5e693b34eb3164fe076d489dc01bea9170843` |
| Merged at | `2023-11-15` |
| Proposal | community `#1486` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

在 PIR 路径下补齐 `fused_elemwise_add_activation` 支持，使相关动转静 build strategy 用例可在 PIR 下运行。

## Why This Sample

- **真实融合算子迁 PIR**：覆盖 infermeta、YAML、translator / adaptor。
- **边界清晰**：聚焦单一 fused op。
- **需 source build**：含 C++ infermeta 与 PIR 注册。

## Files

- `proposal.md`: approved proposal (do not modify in this package PR).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch relative to base.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

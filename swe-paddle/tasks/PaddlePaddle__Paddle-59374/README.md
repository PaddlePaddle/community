# PaddlePaddle__Paddle-59374

This directory converts Paddle PR #59374 into a SWE-Paddle community task candidate (Hackathon No.7: Tensor `apply` / `apply_`).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [59374](https://github.com/PaddlePaddle/Paddle/pull/59374) |
| PR title | `【Hackathon No.7】为 Paddle 新增 apply API -part` |
| Base commit | `4af8ecca447eba12cf57597d95935b0b5f4311b1` |
| Gold endpoint | `9fab1fe754744eaaee8c829b89bbfc9ce230ab19` |
| Merged at | `2023-12-26` |
| Proposal | community `#1483` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

为 Tensor / Variable 新增 `apply` / `apply_`，支持逐元素应用自定义可调用对象，并覆盖动态图与相关静态路径下的基本可用性与错误处理。

## Why This Sample

- **真实 Hackathon API 新增**：覆盖 Python patch 与 C++ pybind。
- **边界清晰**：聚焦 apply 能力，不扩展到其他高阶变换 API。
- **需 source build**：含 eager / PIR pybind 改动。

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

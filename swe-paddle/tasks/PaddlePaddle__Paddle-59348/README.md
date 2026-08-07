# PaddlePaddle__Paddle-59348

This directory converts Paddle PR #59348 into a SWE-Paddle community task candidate (PIR: add `sequence_mask`).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [59348](https://github.com/PaddlePaddle/Paddle/pull/59348) |
| PR title | `【PIR】add sequence_mask in pir` |
| Base commit | `1001b3234973fb1fd2d6ede7afe918c82c792d66` |
| Gold endpoint | `669a3007e45b0b9f4600faa0a0ee3ff51fe90af3` |
| Merged at | `2023-12-08` |
| Proposal | community `#1484` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

在 PIR 路径下补齐 `sequence_mask`，使相关 sequence 用例可在 PIR 下正确运行。

## Why This Sample

- **真实算子迁 PIR**：覆盖 YAML / compat、infermeta、CPU/GPU kernel。
- **边界清晰**：聚焦 `sequence_mask`。
- **需 source build**：含 C++ / CUDA kernel 改动。

## Files

- `proposal.md`: approved proposal (do not modify in this package PR; merge proposal first).
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch relative to base.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: base commit and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

# PaddlePaddle__Paddle-57741

This directory converts Paddle PR #57741 into a SWE-Paddle community task candidate (PIR: add `memcpy`).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [57741](https://github.com/PaddlePaddle/Paddle/pull/57741) |
| PR title | `【PIR】add memcpy in PIR` |
| Base commit | `f984ed1a56960aeee0059c67b965406984565356` |
| Gold endpoint | `4288e25e07895e2fd9985b7a2ec94baedac39159` |
| Merged at | `2023-10-23` |
| Proposal | community `#1487` |
| Task type | `feature_enhancement` |
| Resource | CPU (source build required) |

## Summary

在 PIR 路径下补齐 `memcpy` 算子支持，使动转静场景中的 Tensor 设备间拷贝用例可在 PIR 下正确运行。

## Why This Sample

- **真实 PIR 迁移**：来自已合入的框架算子迁 PIR PR。
- **基础算子路径**：覆盖 YAML / compat、kernel pass 与动转静测试开启。
- **边界清晰**：目标集中在 `memcpy`；不扩展到其他设备管理 API。
- **需 source build**：含 C++ / YAML / pass，不能只靠 wheel overlay。

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

Expected: with `tests/test.patch` on `base_commit`, PIR memcpy related cases fail/error.
After `solution/code.patch` and rebuild, target cases pass.

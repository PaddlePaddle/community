# PaddlePaddle__Paddle-59021

This directory converts Paddle PR #59021 into a SWE-Paddle community task candidate (PIR: fix `test_len` / SelectedRows path and open fuse elewise PIR coverage).

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Primary PR | [59021](https://github.com/PaddlePaddle/Paddle/pull/59021) |
| PR title | `【PIR】Fix test_len` |
| Base commit | `a53f40972d9dea85b44e6eae288f14c1bd01e3a7` |
| Gold endpoint | `3af9eb7eb21f80e81f3573c427feeebbd621a72a` |
| Merged at | `2023-11-27` |
| Proposal | community `#1485` |
| Task type | `bug_fix` |
| Resource | CPU (source build required) |

## Summary

修复 PIR 下 SelectedRows 相关 `len` 路径，并开放 fuse elewise add activation 的 PIR 覆盖测试。

## Why This Sample

- **真实 PIR 兼容修复**：针对 SelectedRows / shape / len 执行链缺口。
- **边界清晰**：聚焦该兼容问题与对应 PIR 覆盖开启。
- **需 source build**：含 C++ adaptor / utils 与 YAML。

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

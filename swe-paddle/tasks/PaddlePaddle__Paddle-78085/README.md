# PaddlePaddle__Paddle-78085

This directory converts Paddle PR #78085 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | [77735](https://github.com/PaddlePaddle/Paddle/issues/77735) |
| PR | [78085](https://github.com/PaddlePaddle/Paddle/pull/78085) |
| PR title | `fix: Implement RestrictedUnpickler to prevent RCE via pickle deserialization (CWE-502)` |
| Base commit | `bb769229044ec937446dbdb55e04ac22d83b313b` |
| Merged at | `2026-03-13T14:57:12+08:00` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

限制模型文件反序列化时可恢复的对象，阻止危险 Pickle 数据触发任意 Python 调用，同时保持正常模型参数和 state dict 的加载兼容性。

## Why This Is A Good SWE-Paddle Candidate

- 任务来自已合入的真实安全缺陷修复，用户可观察行为和风险边界明确。
- production patch 横跨 11 个 Python 文件，覆盖公共加载接口、模型恢复和 checkpoint 相关路径。
- 原 PR 新增 309 行、20 个行为测试，同时验证正常数据兼容性和危险对象拦截。
- 验证可在 CPU 上完成，不依赖 Torch、GPU、CUDA、网络或外部数据集。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold production patch from the merged PR.
- `tests/test.patch`: exact test patch from the merged PR.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Applying `tests/test.patch` to `base_commit` should fail because secure model deserialization is unavailable. Applying both patches should make all 20 original PR tests pass.

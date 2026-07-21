# PaddlePaddle__Paddle-78104

This directory converts Paddle PR #78104 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78104](https://github.com/PaddlePaddle/Paddle/pull/78104) |
| PR title | `[Bugfix] paddle.cuda.device(tensor.place)` |
| Base commit | `d85ad0fca9513ff7d1f0a552649f9136e94cf2a5` |
| Merged at | `2026-03-03` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 device context 接收 Tensor 返回的 generic `Place` 时，无法按其真实 device type 和 device id 转换为对应 concrete Place 的问题。

## Why This Is A Good SWE-Paddle Candidate

- It covers a user-visible device-context regression with a small production change.
- It distinguishes a generic `core.Place` wrapper from concrete CPU, CUDA, XPU, and custom Place objects.
- It preserves existing string-device and concrete-Place behavior.
- The upstream unit-test change is excluded from `solution/code.patch`; this task uses an independent CPU-only verifier.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve string and concrete-Place behavior while failing generic Place conversion; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

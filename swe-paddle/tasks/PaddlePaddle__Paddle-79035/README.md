# PaddlePaddle__Paddle-79035

This directory converts Paddle PR #79035 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79035](https://github.com/PaddlePaddle/Paddle/pull/79035) |
| PR title | `[API Compatibility] Add aliases for apis in paddle.optimizer.lr` |
| Base commit | `e55b609b31d6a00ab35d8fd6e651b2106319ba0d` |
| Merged at | `2026-05-22` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

为 `paddle.optim` 补齐 `lr_scheduler` 兼容命名空间，使用户可以使用与常见 scheduler 命名一致的公开别名，同时保持原有 `paddle.optimizer.lr` 行为不变。

## Why This Is A Good SWE-Paddle Candidate

- 用户可观察契约明确：兼容 namespace 与公开 scheduler 名称能否正常导入和使用。
- production change 范围小且边界清楚，仅涉及 `paddle.optim` 的公开 Python API 暴露。
- 可通过 CPU-only source overlay 稳定验证，无需编译 Paddle、GPU、网络或外部数据集。
- Base 上缺失兼容 namespace，可构造直接相关且确定性的 F2P；原有 namespace 可作为 P2P。

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch for the production files from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

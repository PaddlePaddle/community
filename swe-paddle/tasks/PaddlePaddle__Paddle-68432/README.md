# PaddlePaddle__Paddle-68432

This directory converts Paddle PR #68432 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [68432](https://github.com/PaddlePaddle/Paddle/pull/68432) |
| PR title | 【Hackathon 7th No.18】为稀疏计算添加复数支持2 -part |
| Base commit | `979489bc3280e682f2ce8996d9b0e154ec425a59` |
| Merged at | `2024-09-29T11:58:34Z` |
| Hackathon | `7th` task `18` |
| Track | `C` (feature_or_api) |
| Task type | `feature_enhancement` |
| Resource | CPU10 pilot / CPU |

## Summary

为稀疏计算 multiply_coo_coo/multiply_csr_csr/divide_coo_coo/divide_csr_csr 添加复数支持

## Why This Is The Starter Example

This sample was selected from the CPU10 pilot because Claude Opus 4.8 passed it while ERNIE 5.1 failed. It is also a better first SWE-Paddle example than a simple API boundary fix:

- It touches sparse CPU kernels and Python sparse API behavior.
- It requires complex dtype registration.
- It requires correct complex autograd formulas, including conjugation in multiply/divide gradients.
- In the pilot traces, weaker solutions found the dtype registration issue but missed the complex gradient formula issue.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `swe_paddle.yaml`: structured metadata and verification entrypoint.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `review/author_checklist.md`: contributor and reviewer checklist.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

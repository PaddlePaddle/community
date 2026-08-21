# PaddlePaddle__Paddle-74491

This directory converts Paddle PR #74491 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [#74491](https://github.com/PaddlePaddle/Paddle/pull/74491) |
| PR title | [API compatibility] add new API `paddle.Tensor.requires_grad` |
| Base commit | `01666a6667e744874d7f7c379b2649d8bae67f09` |
| Merged at | `2025-08-13` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add a `requires_grad` compatibility property for Paddle Tensor-like values across dynamic, static, and PIR execution modes while keeping existing gradient-control behavior intact.

## Why This Is A Good SWE-Paddle Candidate

- The task comes from a merged Paddle API-compatibility PR with a narrow, well-defined behavior contract.
- The observable behavior can be verified deterministically without a source build, GPU, network access, or distributed runtime.
- The change spans three execution modes, so the task checks consistency rather than a single isolated code path.
- Existing Tensor metadata behavior can be protected independently with a P2P test.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR, limited to production files.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the new `requires_grad` behavior while the P2P remains valid; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

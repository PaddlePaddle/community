# PaddlePaddle__Paddle-78570

This directory converts Paddle PR #78570 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78570](https://github.com/PaddlePaddle/Paddle/pull/78570) |
| PR title | `[API Compatibility] Support arg `closure` for `paddle.optimizer.optimizer.step`` |
| Base commit | `d8f60c6d12d57d653c97a6c9298f0c11b2db9b2a` |
| Merged at | `2026-04-17` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

为 Optimizer、Adam 和 AdamW 的 `step` API 补充可选 `closure` 参数，并保持不传 `closure` 时的既有更新行为。

## Why This Is A Good SWE-Paddle Candidate

- The target is a merged API-compatibility change with a clear user-visible failure before the fix.
- The closure contract covers argument compatibility, return values, gradient enablement, and the parameter-update handoff.
- The behavior can be verified deterministically on CPU without building Paddle from source.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target closure behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

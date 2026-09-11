# PaddlePaddle__Paddle-79657

This directory converts Paddle PR #79657 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79657](https://github.com/PaddlePaddle/Paddle/pull/79657) |
| PR title | `[Operator Mechanism] Fix paddle.compat.min/max gradient indexing` |
| Base commit | `16037ff1effb88625041f9a1c540e8b2af3ab5c1` |
| Gold commit | `20519ee630ec4929776b33341f690adc2fc48001` |
| Merged at | `2026-08-17` |
| Task type | `bug_fix` |
| Resource | Single NVIDIA GPU |

## Summary

Fixed incorrect gradient indexing in `paddle.compat.min/max` on CUDA when `keepdim=False` and the reduction axis is not trailing. The upstream gradient elements were not routed to the correct input positions, producing wrong backward results.

## Why This Is A Good SWE-Paddle Candidate

- It targets a real GPU backward indexing defect in the `paddle.compat` API layer.
- It requires understanding reduction shape alignment, `keepdim` semantics, and CUDA kernel gradient routing.
- The scope is well-bounded: only CUDA backward with `keepdim=False` and non-trailing axes is affected.
- Forward behavior, `keepdim=True`, trailing-axis, and CPU paths are unaffected and must be preserved.

## Verification Matrix

| State | P2P tests | F2P tests |
| --- | --- | --- |
| Base (`16037ff`) | PASS | FAIL |
| Gold (`20519ee`) | PASS | PASS |

- **P2P** (pass-to-pass): existing elementwise min/max, `keepdim=True` backward, trailing-axis backward, forward shape and values.
- **F2P** (fail-to-pass): CUDA backward with `keepdim=False` and non-trailing axes for both `min` and `max`, including positive and negative axis indices with nonuniform upstream gradients.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: independent test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve forward, `keepdim=True`, trailing-axis, and CPU behavior while failing CUDA backward with `keepdim=False` and non-trailing axes; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

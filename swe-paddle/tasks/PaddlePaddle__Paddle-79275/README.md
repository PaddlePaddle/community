# PaddlePaddle__Paddle-79275

This directory converts Paddle PR #79275 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79275](https://github.com/PaddlePaddle/Paddle/pull/79275) |
| PR title | `[API Compatibility] Align torch.nn.attention.flex_attention.or_masks/and_masks` |
| Base commit | `1d14ac949cd00747df9c828537f5fbff51b1f85f` |
| Merged at | `2026-06-12` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add `or_masks` and `and_masks` combination APIs to `paddle.nn.attention.flex_attention` to enable the logical OR/AND combination of multiple attention mask callables, while ensuring consistent boundary conditions and error handling.

## Why This Is A Good SWE-Paddle Candidate

- It adds two user-visible public APIs rather than fixing an internal-only implementation detail.
- The target contract has clear observable behavior for multiple masks, a single mask, empty input, and invalid non-callable input.
- Existing `paddle.nn.attention` exports provide a direct regression boundary for P2P verification.
- The production change is Python-only and can be exercised deterministically with controlled Tensor-like doubles.
- The task does not require GPU, distributed execution, network access, or a Paddle source build.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

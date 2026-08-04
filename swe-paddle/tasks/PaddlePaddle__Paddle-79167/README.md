# PaddlePaddle__Paddle-79167

This directory converts Paddle PR #79167 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79167](https://github.com/PaddlePaddle/Paddle/pull/79167) |
| PR title | `[API Compatibility] Add alias for paddle.random.initial_seed` |
| Base commit | `c34031973911346f8cd98717583577f61adcf0b1` |
| Merged at | `2026-05-29` |
| Task type | `api_addition` |
| Resource | CPU |

## Summary

Add a top-level public alias, `paddle.initial_seed`, for the existing `paddle.random.initial_seed`, and include this name in Paddle's set of public exports.

## Why This Is A Good SWE-Paddle Candidate

- It adds a user-visible public API instead of repairing an internal-only bug.
- The production change is Python-only and limited to Paddle's top-level export surface.
- Independent tests can distinguish Base and Solution without importing a historical Paddle wheel or building native extensions.
- Existing `seed` and `manual_seed` exports provide clear P2P coverage, while alias identity and `__all__` membership provide direct F2P signals.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the new top-level alias behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

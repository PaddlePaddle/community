# PaddlePaddle__Paddle-79161

This directory converts Paddle PR #79161 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79161](https://github.com/PaddlePaddle/Paddle/pull/79161) |
| PR title | `[API Compatibility] Add param alias for paddle.set_rng_state` |
| Base commit | `8dd02b271734f7aae3669fe6dbcbea57d9cc9add` |
| Merged at | `2026-05-28` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Add a compatible alias `new_state` for the existing `state_list` parameter of `paddle.set_rng_state`, while keeping the original calling method unchanged.

## Why This Is A Good SWE-Paddle Candidate

- It adds a user-visible API compatibility capability rather than repairing an internal-only failure.
- The production change is Python-only and limited to one public random-state API.
- Independent behavior tests can distinguish Base and Solution on CPU without source build, GPU, or external data.
- Positional and original keyword calls provide clear P2P coverage, while alias dispatch and conflict handling provide direct F2P signals.

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

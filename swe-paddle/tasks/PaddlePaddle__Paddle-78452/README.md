# PaddlePaddle__Paddle-78452

This directory converts Paddle PR #78452 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78452](https://github.com/PaddlePaddle/Paddle/pull/78452) |
| PR title | `Support loading dataclass objects in paddle.load()` |
| Base commit | `0156c9d3a222adaca16a394826654a9f449d11aa` |
| Gold commit | `362b943a5a2823f9b2d4a2f0ffe2a2cff07789ab` |
| Merged at | `2026-03-25T12:48:48Z` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

Allow `paddle.load()` to restore model data containing ordinary configuration objects and dataclasses while continuing to reject classes that define unsafe pickle hooks.

## Why This Is A Good SWE-Paddle Candidate

- The failure comes from a real model-loading compatibility issue involving configuration objects commonly stored with checkpoints.
- The fix must balance compatibility with the existing restricted-unpickling security boundary.
- The source PR supplies behavioral tests for allowed configuration classes, rejected dangerous classes, and dataclass handling.
- The production change is Python-only and can be verified deterministically on CPU without external data or network access.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only Gold patch.
- `tests/test.patch`: benchmark regression tests adapted from the merged PR test coverage so Base can collect the complete P2P/F2P role set.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to the Base commit should preserve the existing safe-loading and blocking tests while the new safe configuration loading tests fail. Applying both `tests/test.patch` and `solution/code.patch` should make the complete target test file pass.


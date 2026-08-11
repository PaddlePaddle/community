# PaddlePaddle__Paddle-79197

This directory converts Paddle PR #79197 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79197](https://github.com/PaddlePaddle/Paddle/pull/79197) |
| PR title | `[API Compatibility] Support param optimizer for lr_scheduler` |
| Base commit | `06d8af53d39ef6622689bab27e1cd03a2ffab0f3` |
| Merged at | `2026-06-08T06:44:24Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Allow commonly used learning-rate schedulers to accept an existing optimizer directly and automatically associate themselves with that optimizer, while preserving the existing `learning_rate` calling convention.

## Why This Is A Good SWE-Paddle Candidate

* The issue reflects a common scheduler API mismatch encountered when migrating training code, with clear trigger conditions and expected behavior.
* The change covers shared argument-handling logic used by multiple schedulers and cannot be solved through a one-off special case.
* The source PR provides real tests covering positional and keyword arguments, learning-rate updates, and the association between schedulers and optimizers.
* The tests run on CPU without external datasets, network access, or distributed devices.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: production-only gold patch from the merged PR.
- `tests/test.patch`: exact upstream diff for `test/legacy_test/test_lr_scheduler.py`.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail when schedulers receive an optimizer; applying both `tests/test.patch` and `solution/code.patch` should pass the complete upstream test file.

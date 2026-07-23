# PaddlePaddle__Paddle-55890

This directory converts Paddle PR #55890 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [55890](https://github.com/PaddlePaddle/Paddle/pull/55890) |
| PR title | `[BugFix]Fix bug in vpp+ sharding/dp overlap` |
| Base commit | `42ab2c34b3dd76b54b547613e12393413350c285` |
| Merged at | `2023-08-02` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

Fix incorrect gradient communication scheduling when virtual pipeline parallel is combined with sharding/data parallel overlap.

## Why This Is A Good SWE-Paddle Candidate

- It represents a real distributed-training scheduling bug involving pipeline stages, model chunks, and gradient accumulation.
- The production scope is limited to one Python file while the behavioral distinction remains non-trivial.
- Deterministic CPU-only tests can validate communication timing and chunk selection without launching distributed workers.

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

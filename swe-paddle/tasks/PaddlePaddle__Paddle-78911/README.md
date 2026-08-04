# PaddlePaddle__Paddle-78911

This directory converts Paddle PR #78911 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78911](https://github.com/PaddlePaddle/Paddle/pull/78911) |
| PR title | `fix recompute detection bug` |
| Base commit | `6dfc086f8a3c2245ea1d75891386e82aa5721f15` |
| Gold commit | `6d032b39be8994de8e5e5231d2cf2912533fbddf` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

Improve recompute context detection in dynamic graph mode so that `is_in_recompute()` correctly identifies the recompute context during both the forward pass and backward recomputation.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to the base commit should fail on the target behavior; applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

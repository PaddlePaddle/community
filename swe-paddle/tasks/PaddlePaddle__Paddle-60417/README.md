# PaddlePaddle__Paddle-60417

This directory converts Paddle PR #60417 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [60417](https://github.com/PaddlePaddle/Paddle/pull/60417) |
| PR title | `[auto config] Resume from history csv file` |
| Base commit | `e4b39bb56a4e55213383e96daf262f4f72c1811d` |
| Merged at | `2023-12-29T07:40:45Z` |
| Task type | `feature_enhancement` |
| Resource | CPU |

## Summary

Allow distributed auto-tuner runs to reuse completed configurations from a history CSV and continue with only the unfinished work.

## Why This Is A Good SWE-Paddle Candidate

- Interrupted tuning runs are a practical distributed-training problem, and the wasted work is directly observable.
- The change coordinates CSV value restoration, configuration matching, recorder updates, and launch control rather than adding an isolated special case.
- The source PR contains no test changes, so the candidate adds focused behavior tests that execute the checkout's real `AutoTuner` and `launch()` control flow.
- The tests prove that a resumed task does not start a training controller while preserving the existing search path.
- Verification is deterministic on CPU and requires no Paddle wheel overlay, GPU, network, training data, or subprocess timing.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: exact production patch from the merged commit.
- `tests/test.patch`: focused behavior tests for history resume and launch reuse.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve the existing search case but fail the history-resume cases; applying both `tests/test.patch` and `solution/code.patch` should pass all target tests.

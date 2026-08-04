# PaddlePaddle__Paddle-74221

This directory converts Paddle PR #74221 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74221](https://github.com/PaddlePaddle/Paddle/pull/74221) |
| PR title | `[0-size Tensor No.169] Add 0-size Tensor support for paddle.nn.functional.fold` |
| Base commit | `3bfdd753fa54582b0a2ab6b47e4ea8092cec8187` |
| Gold commit | `763c9f253350af557a781c27c40bc1a2b7b350d2` |
| Merged at | `2025-07-26` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python nn.functional API |

## Summary

Fix `paddle.nn.functional.fold` to properly validate and reject 0-size tensor inputs with a clear `AssertionError`.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python nn.functional API and does not require rebuilding C++ kernels.
- The failure is deterministic: the base revision lacks proper validation for 0-size tensor inputs.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch.
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | fold F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

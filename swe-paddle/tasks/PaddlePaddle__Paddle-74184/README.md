# PaddlePaddle__Paddle-74184

This directory converts Paddle PR #74184 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74184](https://github.com/PaddlePaddle/Paddle/pull/74184) |
| PR title | `[0-size Tensor No.114] Add 0-size Tensor support for paddle.linalg.pinv` |
| Base commit | `ac82c42a5c17f1ddd3ac50a28bb8d0ce84acba8e` |
| Gold commit | `ea80fa17d84889795799fa5f868572b24bd8837c` |
| Merged at | `2025-07-24` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python API (linalg) |

## Summary

Fix `paddle.linalg.pinv` to correctly handle 0-size tensors in both `hermitian=False` (SVD path) and `hermitian=True` (eigh path) branches by adding 0-size early-return logic.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is in the Python-level `pinv` API implementation and requires handling 0-size tensor edge cases in both SVD and eigh code paths.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to `_C_ops.max` on empty singular values and missing transpose shortcut for hermitian 0-size input.
- The task has clear regression coverage for existing non-zero-size behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (Python linalg.py changes).
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | pinv F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

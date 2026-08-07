# PaddlePaddle__Paddle-73569

This directory converts Paddle PR #73569 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73569](https://github.com/PaddlePaddle/Paddle/pull/73569) |
| PR title | `[Accuracy diff No.104] Fix accuracy diff for paddle.matmul API` |
| Base commit | `434044ec20095341e74558c4612a0fe62fcc6508` |
| Gold commit | `1399f8e514c134020f260ad3a79bc889dde810b4` |
| Merged at | `2025-06-27` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix accuracy difference for `paddle.matmul` API when `y` is a 1-D tensor and `transpose_y` is True. The issue was that the gradient kernel incorrectly handled the transpose flag for 1-D `y` tensors.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision produces incorrect gradient results when `y` is 1-D and `transpose_y=True`.
- The task has clear regression coverage for existing matmul behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (C++ kernel changes).
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | matmul F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

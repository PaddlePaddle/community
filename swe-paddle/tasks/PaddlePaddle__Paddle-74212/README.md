# PaddlePaddle__Paddle-74212

This directory converts Paddle PR #74212 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [74212](https://github.com/PaddlePaddle/Paddle/pull/74212) |
| PR title | `[0-size Tensor Job2 No.51] Add 0-size Tensor support for paddle.multiplex` |
| Base commit | `0f3860d981460b0b788aa50836a215f59c90e32a` |
| Gold commit | `3e59330aa066d997e24ff6c5c74c19b250fae43d` |
| Merged at | `2025-07-28` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | C++ Operator Kernel |

## Summary

Fix `paddle.multiplex` to correctly handle 0-size tensors in CPU/GPU kernels by adding early-return logic when output numel is 0.

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the C++ operator kernel level and requires rebuilding Paddle from source.
- The failure is deterministic: the base revision fails when processing 0-size tensors due to PADDLE_ENFORCE_GT checks on input numel.
- The task has clear regression coverage for existing non-zero-size behavior.
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

| Revision state | Existing behavior (P2P) | multiplex F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

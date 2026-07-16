# PaddlePaddle__Paddle-78238

This directory converts Paddle PR #78238 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [78238](https://github.com/PaddlePaddle/Paddle/pull/78238) |
| PR title | `[ZeroDim] fix put_along_axis with 0-size indices tensor` |
| Base commit | `ae907b878e91dbabf3582da99f8b05a46b588fc2` |
| Gold commit | `2d26799c4f1a710deb78b7a2182c60eaa0ee7d22` |
| Merged at | `2026-03-20` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python Tensor API |

## Summary

完善 `paddle.put_along_axis` 和 `Tensor.put_along_axis_` 对 0-size `indices` Tensor 的支持。

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python Tensor API and does not require rebuilding C++ kernels.
- The failure is deterministic: the base revision reaches an invalid broadcast when the index tensor is empty.
- The task has clear regression coverage for existing non-empty-index behavior.
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

| Revision state | Existing behavior | Zero-size out-of-place | Zero-size in-place |
| --- | ---: | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS | PASS |

# PaddlePaddle__Paddle-73582

This directory converts Paddle PR #73582 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [73582](https://github.com/PaddlePaddle/Paddle/pull/73582) |
| PR title | `[0-size Tensor Job2 No.12、77] Add 0-size Tensor support for paddle.squeeze/full` |
| Base commit | `ecd685afb0ffc1f509771cd1820254c8b42020ad` |
| Gold commit | `f69b42e57712ab1c68edc071bee41758c27612f7` |
| Merged at | `2025-06-30` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python API (squeeze/full 0-size Tensor support) |

## Summary

Fix `paddle.squeeze` and `paddle.full` to correctly handle 0-size Tensor inputs:
- `paddle.squeeze`: When `axis` is a 0-size Tensor, return `x` unchanged
- `paddle.full`: When `shape` contains 0-size Tensor elements, skip them in conversion

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a merged Paddle bug-fix PR rather than a synthetic issue.
- The target behavior is isolated to the Python API layer and does not require C++ kernel changes.
- The failure is deterministic: the base revision fails when `squeeze` receives a 0-size axis tensor or when `full` receives a shape containing 0-size tensors.
- The task has clear regression coverage for existing squeeze/full behavior.
- The task runs on CPU and does not require distributed execution, external services, or additional datasets.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch (Python API changes).
- `tests/test.patch`: tests exposing the target behavior.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing behavior (P2P) | squeeze/full F2P |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

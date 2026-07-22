# PaddlePaddle__Paddle-79057

This directory converts Paddle PR #79057 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79057](https://github.com/PaddlePaddle/Paddle/pull/79057) |
| PR title | `[Security] Fix RCE vulnerability in RestrictedUnpickler MRO check` |
| Base commit | `f4014bfa7b9acddfcfcaffb57b57b2a5c8fe9e7a` |
| Gold commit | `bf24243bd9e640590362c3f92abd74ca69a3ef30` |
| Merged at | `2026-05-22` |
| Task type | `bug_fix` |
| Resource | CPU |
| Scope | Python secure deserialization |

## Summary

完善 `RestrictedUnpickler` 对 MRO 中 unsafe pickle hooks 的识别，避免因仅检查当前 class 而遗漏 base class 定义的方法。

## Why This Is A Good SWE-Paddle Candidate

- It comes from a merged Paddle security bug-fix PR rather than a synthetic issue.
- The production change is isolated to one Python helper and is small enough for focused review.
- The failure is deterministic and can be reproduced without executing external commands or requiring a network service.
- The tests cover both the class-safety decision and the actual restricted-unpickling path.
- Existing safe user-defined classes and directly dangerous classes remain covered as regression guards.
- The task is CPU-only and requires no Paddle build, GPU, distributed runtime, or external dataset.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold implementation patch from the upstream PR.
- `tests/test.patch`: tests exposing the inherited-hook bypass.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment and reproduction notes.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

| Revision state | Existing policy | Inherited-hook checks |
| --- | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL |
| Base + `tests/test.patch` + `solution/code.patch` | PASS | PASS |

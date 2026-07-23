# PaddlePaddle__Paddle-77150

This directory converts Paddle PR #77150 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [77150](https://github.com/PaddlePaddle/Paddle/pull/77150) |
| PR title | `[Eager] Add missing attribute copy when PyLayer grad node copied` |
| Base commit | `bbc3fbcf1b93bb5fc2f6425ccbcda22816b7c8ab` |
| Merged at | `2025-12-30` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复自定义 `PyLayer` 返回部分不可导输出时，`paddle.grad` 可能因 Eager autograd 节点状态不完整而异常退出的问题。

## Why This Is A Good Candidate

- It is a real framework-level bug from a merged Paddle pull request.
- The failure is externally observable through public `PyLayer` and `paddle.grad` APIs.
- The regression test is small, deterministic, and requires no GPU, network service, model file, or private data.
- A correct solution must preserve copied autograd-node state without changing existing multi-output `PyLayer.backward()` behavior.
- The task has clear F2P and P2P test node IDs in one legacy test file.

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

Expected behavior: applying `tests/test.patch` to `base_commit` should preserve the P2P test while the target F2P test fails or terminates the test process; applying both `tests/test.patch` and `solution/code.patch`, followed by rebuilding the affected Paddle binary, should make both target tests pass.

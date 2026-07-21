# PaddlePaddle__Paddle-65724

This directory converts Paddle PR #65724 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| Issue | [48964](https://github.com/PaddlePaddle/Paddle/issues/48964) |
| PR | [65724](https://github.com/PaddlePaddle/Paddle/pull/65724) |
| PR title | `[Bugfix] fix dataloader when setting persistent_workers=True` |
| Base commit | `9a4caad68bca019e85847eb99da57f060e01caa5` |
| Gold commit | `b7c8439cd489e26c09c1db8b929285a96c64e3ed` |
| Merged at | `2024-07-17` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 `DataLoader` 在 `persistent_workers=True` 且一个 epoch 被提前终止后，复用 workers 进入后续 epoch 时可能因 batch structure metadata 不一致而崩溃的问题。

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a real merged BugFix PR linked to issue #48964.
- The production change is limited to one Python file.
- The failure concerns a practical benchmark workflow: breaking an epoch early while keeping workers persistent.
- The verifier checks producer/consumer handoff and iterator reset behavior instead of matching source text.
- The tests are deterministic and CPU-only; they do not require GPU training or long-running worker processes.
- Existing FIFO structure restoration remains covered by P2P.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: independent regression verifier.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: environment notes for reproduction.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior:

- Applying `tests/test.patch` to `base_commit` keeps ordinary FIFO structure handoff green.
- The producer/consumer handoff test and persistent-worker reset test fail on `base_commit`.
- Applying both `tests/test.patch` and `solution/code.patch` makes all target tests pass.
- The patched production file must match the Git blob from `gold_commit`.

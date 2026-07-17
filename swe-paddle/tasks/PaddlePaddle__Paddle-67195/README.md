# PaddlePaddle__Paddle-67195

This directory converts Paddle PR #67195 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [67195](https://github.com/PaddlePaddle/Paddle/pull/67195) |
| PR title | `[BugFix] Fix pp nan checker before send` |
| Base commit | `87d69ba93e5db77d9c0647d5954bd43a7fcb5ea5` |
| Gold commit | `d02476451b085229af521fa38d58a9f2a5ea1f9e` |
| Merged result | `643e71334cae5d9fd966db90a64aacfcdd87eae4` |
| Merged at | `2024-08-09` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

修复 Pipeline Parallel P2P communication 的 NaN/Inf checker 执行时序，确保 invalid outgoing Tensor 在任何 send/recv operation 启动前被拒绝。

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a real merged Distributed Strategy BugFix PR.
- The production change is limited to one Python file.
- The observable contract is precise: invalid outgoing data must not enter P2P communication.
- The verifier records communication side effects and does not match source text.
- P2P coverage preserves valid send behavior and confirms receive tensors are not treated as outgoing payloads.
- The task runs deterministically on CPU without initializing a distributed process group.

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

- Applying `tests/test.patch` to `base_commit` keeps valid P2P behavior green.
- Invalid outgoing Tensor tests fail on `base_commit` because communication is launched before the error is reported.
- Applying both `tests/test.patch` and `solution/code.patch` makes all target tests pass.
- The patched production file must match the Git blob from `gold_commit`.

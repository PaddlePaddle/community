# PaddlePaddle__Paddle-70469

This directory converts Paddle PR #70469 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [70469](https://github.com/PaddlePaddle/Paddle/pull/70469) |
| PR title | `[BugFix] Fall back fused dropout add` |
| Base commit | `052874d0c95fe9bcae0c3e0ac60d857b238e6d70` |
| Gold commit | `5cc69a1b9adbb4b3f3ce30312c0b031be503dff4` |
| Merged result | `2d1fe5871551c28d92c307f1bbd6d6b12aeeb8c2` |
| Merged at | `2024-12-27` |
| Task type | `bug_fix` |
| Resource | CPU |

## Summary

为规避 `fused_dropout_add` fused execution path 中已知的 precision issue，暂时回退到标准 `dropout + add` 实现，并提供一次性 warning。

## Why This Is A Good SWE-Paddle Candidate

- It is derived from a real merged Distributed Strategy BugFix PR.
- The production change is limited to one Python file.
- The expected behavior is observable through output semantics, dispatch side effects, and warning behavior.
- The verifier executes the checked-out implementation with controlled operator doubles instead of matching source text.
- P2P coverage preserves the `p == 0` fast-result behavior.
- The task runs deterministically on CPU without invoking the GPU-only fused kernel.

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

- Applying `tests/test.patch` to `base_commit` keeps the `p == 0` P2P green.
- Training fallback and warning-semantics tests fail on `base_commit`.
- Applying both `tests/test.patch` and `solution/code.patch` makes all target tests pass.
- The patched production file must match the Git blob from `gold_commit`.

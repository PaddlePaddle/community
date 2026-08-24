# PaddlePaddle__Paddle-76522

This directory converts Paddle PR #76522 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [76522](https://github.com/PaddlePaddle/Paddle/pull/76522) |
| PR title | `[Compat] Auto register compat module overrides when enable torch proxy` |
| Base commit | `b5efb98a163a2be2505e72266841e64b88254a8a` |
| Gold commit | `20d9626540daf86096cc5bd11c9b84b398ce7138` |
| Merged at | `2025-11-24` |
| Task type | `refactor` / `compatibility_improvement` |
| Resource | CPU |

## Summary

Improve the torch proxy compatibility layer so enabling the proxy automatically exposes public `paddle.compat` overrides through the corresponding `torch` namespace, including overrides located in nested submodules.

## Why This Is A Good SWE-Paddle Candidate

- It comes from a merged Paddle compatibility PR and represents a refactoring-oriented task rather than a conventional bug fix.
- The expected behavior is externally observable through the proxy namespace instead of implementation-specific source structure.
- The task covers both automatic override registration and nested module proxying while preserving existing fallback behavior.
- The production change is Python-only and can be verified deterministically on CPU without building Paddle native extensions.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained task statement.
- `solution/code.patch`: production-only Gold patch.
- `tests/test.patch`: independent regression tests.
- `tests/test.sh`: minimal target test command.
- `environment/README.md`: reproduction environment notes.
- `README.md`: task overview and verification entrypoint.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with only `tests/test.patch` applied to the Base commit, the existing proxy fallback test passes while the new compatibility-override tests fail. After applying `solution/code.patch`, all target tests pass.

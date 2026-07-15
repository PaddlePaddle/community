# PaddlePaddle__Paddle-76259

This directory converts Paddle PR #76259 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [76259](https://github.com/PaddlePaddle/Paddle/pull/76259) |
| PR title | `[Bug fixes] Fix Windows UTF-8 path support for Paddle Inference model/json files` |
| Base commit | `fcf3b100085b10efed4c1fb8880b1df1fd5241d6` |
| Gold commit | `59670139385c14790ecd3c764d819eff63821978` |
| Merged at | `2025-11-06` |
| Track | `bug_fix` |
| Task type | `windows_inference_path` |
| Resource | Windows CPU / MSVC |

## Summary

Paddle Inference accepts model and configuration paths as UTF-8 strings. On Windows, existing files or directories whose path contains non-ASCII characters can be misclassified as missing when narrow filesystem APIs are used. This task asks the agent to preserve ordinary ASCII-path behavior while fixing UTF-8 non-ASCII path handling on Windows.

## Files

- `proposal.md`: candidate proposal for maintainer triage.
- `instruction.md`: self-contained problem statement for the coding agent.
- `environment/README.md`: environment notes for reproduction.
- `solution/code.patch`: gold patch from the merged PR.
- `tests/test.patch`: test patch exposing the target behavior.
- `tests/test.sh`: minimal target test command.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: applying `tests/test.patch` to `base_commit` should fail on the UTF-8 Windows path behavior while preserving the ASCII-path guard. Applying both `tests/test.patch` and `solution/code.patch` should pass the target tests.

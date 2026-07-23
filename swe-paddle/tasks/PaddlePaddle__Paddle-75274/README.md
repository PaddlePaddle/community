# PaddlePaddle__Paddle-75274

This directory converts Paddle PR #75274 into a SWE-Paddle task candidate.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| Upstream PR | [#75274](https://github.com/PaddlePaddle/Paddle/pull/75274) |
| PR title | `【UnitTestFix No.10】fix test_normal.py` |
| Base commit | `5d1846ae16a8cecad8545b83d53a56e1a0eebe73` |
| Gold commit | `834096be299fe1a032989a98a98572900a760a60` |
| Merged at | `2025-09-22` |
| Task type | `bug_fix` |
| Subtype | `unit_test_maintenance` |
| Resource | CPU |
| Target scope | One Python test file |

## Summary

修复 `normal` 算子单元测试对全局静态图/动态图状态的隐式依赖，使静态测试路径在不同执行顺序下都能正确构图和运行。

## Why This Is A Good SWE-Paddle Candidate

- The task is derived from a real merged bug-fix PR.
- The broken artifact is the unit test implementation itself, so the gold patch legitimately modifies a test file.
- The production scope is limited to one Python test module.
- The verifier is independent of the repaired file and checks observable execution behavior.
- Base failures reproduce the reported graph-mode errors instead of import, path, GPU, or network failures.
- The task is deterministic and CPU-only.

## Files

- `proposal.md`: candidate rationale and validation plan.
- `instruction.md`: self-contained issue statement.
- `environment/README.md`: runtime and reproduction notes.
- `solution/code.patch`: exact upstream test-maintenance change.
- `tests/test.patch`: independent benchmark verifier.
- `tests/test.sh`: minimal test command.

## Validation Matrix

| Revision state | Dygraph P2P | Scalar static F2P | Tensor static F2P |
| --- | ---: | ---: | ---: |
| Base + `tests/test.patch` | PASS | FAIL | FAIL |
| Base + test patch + `solution/code.patch` | PASS | PASS | PASS |

## Run

```bash
bash tests/test.sh
```

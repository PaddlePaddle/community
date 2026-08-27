# PaddlePaddle__Paddle-78342

This directory packages Paddle PR #78342 as a SWE-Paddle task.

## Source Metadata

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| Pull request | [#78342](https://github.com/PaddlePaddle/Paddle/pull/78342) |
| Title | `[API Compatibility] add new api paddle._assert -part` |
| Base commit | `fa323f323bb35359c9d4ba77763834fee82a87b4` |
| Gold commit | `f92a35feea4acf62b2df2259ae491b992851f854` |
| Resource scope | CPU |
| Patch scope | Python API and legacy unittest coverage |

## Behavior Summary

The task adds the public `paddle._assert(condition, message="")` API with a calling convention compatible with `torch._assert`. In dynamic mode, truthy Python and Tensor conditions return normally, while falsy conditions raise `AssertionError` with the supplied message or an empty default message. Positional, keyword, and mixed argument forms are supported.

In static mode, a Tensor condition contributes an executable assertion to the program rather than being checked while the program is constructed. The existing compatibility APIs in the same regression file must continue to pass.

## Test Classification

- **F2P:** the seven `TestAssertAPI` cases added to `test/legacy_test/test_api_compatibility_part2.py` cover truthy and falsy Python conditions, truthy and falsy Tensor conditions, the default message, compatible calling forms, and static-graph execution.
- **P2P:** `test/legacy_test/test_assert_close.py::TestAssertClose` guards `paddle.testing.assert_close`, which lives in the same comparison module and package export list that the solution patch edits, and passes before and after the solution.

## Artifact Map

- `proposal.md`: approved source proposal; preserved unchanged.
- `instruction.md`: self-contained observable public API requirements.
- `solution/code.patch`: exact base-to-gold diff for the three production files.
- `tests/test.patch`: exact base-to-gold diff for the legacy unittest file.
- `tests/test.sh`: narrowed pytest wrapper with explicit P2P and F2P selections.
- `environment/README.md`: revisions, runtime assumptions, patch order, and verification guidance.

## Verification Summary

- Both patches are exported from the exact base-to-gold Git diff with the required production/test boundary.
- Packaging checks cover patch whitespace, shell syntax, isolated test-first application order, byte-for-byte gold equivalence, and Python syntax.
- The target class was verified fail-before/pass-after and the P2P selection verified passing in both states, with the pass-after run performed on a runtime built from the exact base commit. Per-runtime results and why the wrapper selects a class instead of the whole compatibility file are recorded in `environment/README.md`.

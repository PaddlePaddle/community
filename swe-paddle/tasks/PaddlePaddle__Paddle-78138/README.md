# PaddlePaddle__Paddle-78138

This directory converts Paddle PR #78138 into a focused SWE-Paddle community task.

## Source

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| PR | [#78138](https://github.com/PaddlePaddle/Paddle/pull/78138) |
| PR title | `[API Compatibility] cpp sink paddle.nn.functional.pixel_shuffle -part` |
| Base commit | `555b4a95615a35b301f348e081e56435a6d75da6` |
| Squash commit | `01b7cdd95813a88bca9569f55328c4f6f0e675cb` |
| Merged at | `2026-03-11T02:13:30Z` |
| Task type | `feature_enhancement` |
| Resource | CPU with a mandatory source rebuild |

## Behavioral Summary

The task moves `paddle.nn.functional.pixel_shuffle` to the generated C++-sink API path while preserving its public behavior. Calls using a positional input Tensor or the existing `x=` keyword remain valid, and the PyTorch-compatible `input=` alias is added. Positional, `x=`, and `input=` forms must produce identical results in dynamic and static graph modes.

Existing pixel-shuffle behavior remains unchanged, including output values and shape, `NCHW`/`NHWC` layouts, supported dtypes, gradients, and argument validation.

The squash commit also reorganized `paddle.unique` documentation and tests. Those independent changes are intentionally excluded from this focused task.

## Artifacts

- `proposal.md`: approved task proposal and scope.
- `instruction.md`: self-contained observable requirements for the solver.
- `solution/code.patch`: exact base-to-gold changes for the five task-relevant production files.
- `tests/test.patch`: focused compatibility import and `TestPixelShuffleAPI_Compatibility` class only.
- `tests/test.sh`: strict F2P compatibility test followed by the complete P2P pixel-shuffle operator module.
- `environment/README.md`: base revision, build requirements, patch order, commands, and risks.

## Verification Overview

Package validation is performed from a clean snapshot at the exact base commit. It checks artifact presence, shell syntax, patch whitespace and application order, byte equality of all five production results against the squash commit, focused test equality against gold, exclusion of the unrelated `unique` changes, and Python syntax.

Runtime verification requires a Paddle package built from the patched source tree after code generation. A prebuilt wheel or a build from another revision is not sufficient for the generated binding changes.

Local package-level validation passed. Runtime target tests were not run because the available Paddle installation reports commit `d9242d558766b5d24fa3231a798864c37cdd5cda`, not an exact build of this task's patched base; source-build execution therefore remains unverified locally.

From the root of a rebuilt Paddle checkout with both patches applied:

```bash
bash tests/test.sh
```

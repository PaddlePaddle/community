# PaddlePaddle__Paddle-78220

This directory packages Paddle PR #78220 as a SWE-Paddle task.

## Source Metadata

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| Pull request | [#78220](https://github.com/PaddlePaddle/Paddle/pull/78220) |
| Title | `[API Compatibility] add paddle.compat.nn.functional.log_softmax -part` |
| Base commit | `56be465924264e1251cf127dbff56d17a7554d01` |
| Gold commit | `bfe91230d558176d2d932b50953cb7b4391065d1` |
| Resource scope | CPU |
| Patch scope | Python API and legacy unittest coverage |

## Behavior Summary

The task adds PyTorch-compatible `log_softmax` access and parameter semantics while preserving Paddle's existing behavior. The operation is available through `paddle.nn.functional`, `paddle`, `Tensor`, `paddle.special`, and `paddle.compat.nn.functional`; when equivalent dimensions are explicitly specified, the public routes produce numerically consistent results. Their omitted-dimension defaults are intentionally not identical: the standard Paddle functional API keeps `axis=-1`, while compatibility-style routes use the rank-dependent `dim=None` rule.

The standard Paddle API continues to accept `x`, `axis`, and `name`, and also accepts the `input` and `dim` aliases plus `dtype` and `out`. The compatibility API accepts `input`, `dim`, `dtype`, and `out`, defaults `dim=None` by input rank (0D/1D/3D to dimension 0, otherwise dimension 1), ignores an integer `_stacklevel`, and rejects Paddle-only compatibility keywords. Alias conflicts are rejected. Existing numerical, gradient, static-graph, and PIR behavior remains covered by the regression tests.

## Test Classification

- **F2P:** `test/legacy_test/test_compat_log_softmax.py` exercises the new public access paths, aliases, default dimensions, dtype conversion, output tensors, `_stacklevel`, and strict keyword behavior.
- **P2P:** `test/legacy_test/test_log_softmax.py` retains the existing operator/API regression coverage and adds `TestLogSoftmaxParamAlias` for the standard API aliases and output parameter.

## Artifact Map

- `proposal.md`: approved source proposal with an added clarification of the rank-dependent default-dimension behavior.
- `instruction.md`: self-contained observable requirements for the implementation.
- `solution/code.patch`: exact base-to-gold diff for the six production files.
- `tests/test.patch`: exact base-to-gold diff for the two test files.
- `tests/test.sh`: strict direct-Python target wrapper.
- `environment/README.md`: revisions, runtime/build assumptions, patch order, and verification commands.

## Verification Summary

- The two patches were exported from the exact base-to-gold Git diff with the requested file boundaries.
- Patch whitespace, shell syntax, isolated application order, byte-for-byte gold equivalence, and Python syntax are verified by the packaging checks.
- Runtime execution is best-effort because the available local Paddle build comes from ancestor commit `555b4a95615a35b301f348e081e56435a6d75da6`, rather than the exact historical base. Any runtime limitation is recorded in `environment/README.md`.

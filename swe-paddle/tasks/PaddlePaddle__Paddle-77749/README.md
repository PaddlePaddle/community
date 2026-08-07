# PaddlePaddle__Paddle-77749

This directory packages Paddle PR #77749 as a SWE-Paddle task.

## Source Metadata

| Field | Value |
| --- | --- |
| Repository | `PaddlePaddle/Paddle` |
| Pull request | [#77749](https://github.com/PaddlePaddle/Paddle/pull/77749) |
| Title | `[API Compatibility] implement nn.utils.rnn.pad_sequence and unpad_sequence` |
| Base commit | `ea0f979936ab101a91a8739bdb0a528b8df42ef7` |
| Gold commit | `7c19c94684c0e93b6d5f2b288d34d2a61e39b02a` |
| Resource scope | CPU |
| Patch scope | Python API and legacy unittest coverage |

## Behavior Summary

The task adds `paddle.nn.utils.rnn.pad_sequence` and `paddle.nn.utils.rnn.unpad_sequence`, and exports both operations through `paddle.nn.utils`.

`pad_sequence` accepts a list or tuple of variable-length Tensors, pads them to the longest sequence, supports time-major and batch-major layouts, configurable left or right padding, custom padding values, scalar trailing dimensions, multidimensional trailing dimensions, and integer dtypes. Invalid sequence containers and padding sides are rejected.

`unpad_sequence` restores a list of variable-length Tensors from a padded Tensor and a lengths Tensor in either supported layout. Pad/unpad round trips preserve values, shapes, and tested dtypes.

## Test Classification

- **F2P:** all 21 cases in the new `test/legacy_test/test_rnn_utils.py` file. They cannot import the new module on the exact base and pass after the solution.
- **P2P:** the exact gold test patch adds no independent P2P cases because the entire target file is new. Existing Paddle utility and Tensor behavior remains an external regression expectation.

## Artifact Map

- `proposal.md`: approved source proposal; preserved unchanged.
- `instruction.md`: self-contained observable requirements for the public APIs.
- `solution/code.patch`: exact base-to-gold diff for the two production files.
- `tests/test.patch`: exact base-to-gold diff adding the unittest file.
- `tests/test.sh`: strict direct-Python unittest wrapper.
- `environment/README.md`: revisions, runtime assumptions, patch order, and verification guidance.

## Verification Summary

- Both patches are exported from the exact base-to-gold Git diff with the requested production/test boundary.
- Packaging checks cover patch whitespace, shell syntax, isolated test-first application order, byte-for-byte gold equivalence, and Python syntax.
- Runtime execution is performed with isolated exact-source overlays when the exact historical native build is unavailable; the runtime commit and results are recorded in `environment/README.md`.

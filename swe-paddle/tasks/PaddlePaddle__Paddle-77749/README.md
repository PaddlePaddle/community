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

- **F2P:** all 21 cases in the new `test/legacy_test/test_rnn_utils.py` file. The target APIs are imported inside the test bodies, so the file is collected on the exact base and every case fails there with `ModuleNotFoundError`; all 21 pass after the solution.
- **P2P:** `test/legacy_test/test_rnn_cell_api.py::TestRnnUtil::test_case`, an untouched RNN utility regression guard for `paddle.utils.map_structure` / `paddle.utils.assert_same_structure`. The gold diff does not modify that file, and the node passes on both the exact base and gold revisions.

## Artifact Map

- `proposal.md`: approved source proposal; preserved unchanged.
- `instruction.md`: self-contained observable requirements for the public APIs.
- `solution/code.patch`: exact base-to-gold diff for the two production files.
- `tests/test.patch`: adds the gold unittest file, with the two target API imports deferred into thin wrappers so the file is collectible on the base revision. Every case body is unchanged from gold.
- `tests/test.sh`: wrapper that runs the existing RNN utility P2P node before the 21 new F2P API cases, both under `pytest` so every case reports as its own node.
- `environment/README.md`: revisions, runtime assumptions, patch order, and verification guidance.

## Verification Summary

- `solution/code.patch` is the exact base-to-gold Git diff for the two production files. `tests/test.patch` is the gold test file with only the module-level target import replaced by per-call imports, so the 21 cases are collected and fail individually on the base revision.
- Packaging checks cover patch whitespace, shell syntax, isolated test-first application order, byte-for-byte gold equivalence for the production files, and Python syntax.
- The selected `TestRnnUtil` P2P node is byte-for-byte unchanged between the base and gold revisions and passed repeated runs on both isolated overlays; the base target collected all 21 cases and failed all 21 inside their bodies, and the gold target passed all 21. Runtime details are recorded in `environment/README.md`.

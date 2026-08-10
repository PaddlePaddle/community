# Proposal

## Problem

`paddle.hsplit`, `paddle.dsplit`, and `paddle.vsplit` expose Paddle-native argument names `x` and `num_or_indices`. Code written with PyTorch-style keyword names such as `input`, `indices`, and `sections` cannot call these APIs directly even though the underlying split semantics already match.

## Expected behavior

Each API should accept the following equivalent parameter names:

- `x` / `input`
- `num_or_indices` / `indices` / `sections`

Existing positional calls and Paddle-native keyword calls must continue to produce identical results.

## Verification

The task uses the real compatibility tests added by the source PR in `test/legacy_test/test_api_compatibility.py`.

F2P coverage runs the PR's `TestHsplitAPI`, `TestDsplitAPI`, and `TestVsplitAPI` dynamic-mode compatibility methods. These tests exercise real Paddle Tensor inputs, the original parameter names, the new aliases, Tensor methods, and NumPy reference results.

P2P coverage separately verifies that the existing positional and Paddle-native keyword forms remain unchanged.

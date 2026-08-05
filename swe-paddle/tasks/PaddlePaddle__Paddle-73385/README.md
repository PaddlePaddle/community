# SWE-Paddle Task: PaddlePaddle__Paddle-73385

## Task Overview

This task package corresponds to [Paddle PR #73385](https://github.com/PaddlePaddle/Paddle/pull/73385), which adds 0-size Tensor support for `paddle.linalg.svdvals` and `paddle.linalg.eigvals`.

## Source Information

| Field | Value |
|-------|-------|
| Instance ID | `PaddlePaddle__Paddle-73385` |
| PR Link | https://github.com/PaddlePaddle/Paddle/pull/73385 |
| PR Title | [0-size Tensor Job2 No.42、45] Add 0-size Tensor support for paddle.linalg.svdvals |
| Base Commit | `d48d3a3c1726f1a6f3b4654e9283585006ef5478` |
| Gold Commit | `e6012dd42105045aea42610fc482b54e14210510` |
| Merged At | 2025-06-18 |
| Task Type | `bug_fix` |
| Execution Backend | `cpu` |
| Device Scope | `cpu_only` |
| Module Tags | `[operator_kernel, svdvals, eigvals, 0-size_tensor, cpu_kernel]` |

## Problem Description

When the input to `paddle.linalg.svdvals` or `paddle.linalg.eigvals` is a 0-size Tensor, the current CPU kernel implementation crashes or produces incorrect results because it does not handle the 0-size boundary case.

## Solution Summary

Add early-return logic in the CPU kernels when the output tensor has 0 elements:
- `eigvals_kernel.cc`: Add `if (out && out->numel() == 0) return;`
- `svdvals_kernel.cc`: Add `if (S && S->numel() == 0) { Alloc; return; }`
- `svdvals_grad_kernel_impl.h`: Add `if (x_grad && x_grad->numel() == 0) { Alloc; return; }`

## Files Modified

### Code Changes (solution/code.patch)
- `paddle/phi/kernels/cpu/eigvals_kernel.cc`
- `paddle/phi/kernels/cpu/svdvals_kernel.cc`
- `paddle/phi/kernels/impl/svdvals_grad_kernel_impl.h`

### Test Changes (tests/test.patch)
- `test/legacy_test/test_eigvals_op.py`: Add `TestEigvalsOp_ZeroSize` and `TestEigvalsOp_ZeroSize2`
- `test/legacy_test/test_svdvals_op.py`: Add `TestSvdvalsOp_ZeroSize`, remove old empty tensor exception test

## Verification

Run the test script:
```bash
bash tests/test.sh
```

Expected results:
- **Before fix**: F2P tests fail (kernel crashes or produces errors on 0-size input)
- **After fix**: All P2P and F2P tests pass

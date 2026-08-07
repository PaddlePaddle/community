# PaddlePaddle__Paddle-79310

This directory converts Paddle PR #79310 into a SWE-Paddle community task candidate.

## Source

| Field | Value |
| --- | --- |
| Repo | `PaddlePaddle/Paddle` |
| PR | [79310](https://github.com/PaddlePaddle/Paddle/pull/79310) |
| PR title | `[API Compatibility] Add paddle.nn.init.sparse_()` |
| Base commit | `14f2f9df49bd9bd7fd94eb9cdef850c581243784` |
| Merged at | `2026-06-21` |
| Task type | `api_addition` / `feature_enhancement` |
| Resource | CPU |

## Summary

Added `paddle.nn.init.sparse_()`, a 2D sparse initialization API, 
and unified the in-place return semantics of initializer helpers in dynamic graph mode: 
the input Tensor itself is returned after initialization.

## Why This Is A Good SWE-Paddle Candidate

- It adds a user-visible initializer API rather than fixing an internal-only bug.
- The sparse initializer has a concrete numerical contract: initialize values, then zero a defined fraction of every column.
- The API has an explicit dimensionality boundary that can be tested independently.
- The Gold change also makes existing in-place initializer helpers return the input Tensor in dygraph mode, giving an observable compatibility contract beyond symbol existence.
- The target behavior can be tested deterministically from the checkout's real Python function bodies without CUDA, native kernels, or a Paddle source build.

## Patch Boundary

`solution/code.patch` contains only the production file changed by the Gold commit:

- `python/paddle/nn/init.py`

The original PR change to `test/legacy_test/test_nn_init_function.py` is intentionally excluded. Independent benchmark tests are supplied only through `tests/test.patch`.

The packaged `solution/code.patch` is a bootstrap representation. During cross validation it is replaced by the exact production-only Gold diff generated from local Git objects.

## Verification

```bash
bash tests/test.sh
```

Expected behavior: with only `tests/test.patch`, the existing non-dygraph `normal_` path remains valid while the new dygraph-return, `eye_` return, and `sparse_` behaviors fail. After applying the exact Gold production patch, all target tests should pass.

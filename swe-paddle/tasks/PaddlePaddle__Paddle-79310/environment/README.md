# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `14f2f9df49bd9bd7fd94eb9cdef850c581243784`
- Resource: CPU
- GPU required: no
- Build path: Python-only checkout source with AST overlays for the target initializer functions; no Paddle source build is required.

## Why GPU Is Not Required

The benchmark executes the real Python bodies of the target initializer functions in a controlled namespace. Tensor mutation, random index generation, no-grad context behavior, and initializer calls are represented by deterministic doubles. CUDA kernels, GPU allocation, NCCL, and device synchronization are not used.

A GPU-enabled machine is compatible with the task, but the benchmark itself is CPU-only.

## Gold Patch Boundary

`solution/code.patch` must contain only:

- `python/paddle/nn/init.py`

Do not include the original PR modification to:

- `test/legacy_test/test_nn_init_function.py`

The verifier checks the complete Gold changed-file scope separately and generates the final solution patch from Git objects using only the production path.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the existing P2P should pass and the new initializer behaviors should fail.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass after the Gold production patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

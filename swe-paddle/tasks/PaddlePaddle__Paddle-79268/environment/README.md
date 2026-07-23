# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `722421e3a49eadf5ea774639c3d8147aced333ce`
- Resource: CPU
- GPU required: no
- Build path: Python-only checkout source with AST overlays for the sampler classes and public data-module exports; no Paddle source build is required.

## Why GPU Is Not Required

The target behavior consists of Python-side index generation, deterministic NumPy shuffling, constructor argument forwarding, and public module exports. The tests explicitly provide `num_replicas` and `rank`, so they do not initialize a process group or execute collective communication.

A GPU-enabled environment is compatible with this task, but CUDA kernels, NCCL, device allocation, and GPU synchronization are not used.

## Gold Patch Boundary

`solution/code.patch` must contain only:

- `python/paddle/io/dataloader/batch_sampler.py`
- `python/paddle/utils/data/__init__.py`
- `python/paddle/utils/data/distributed.py`

Do not include the original PR modifications to:

- `test/legacy_test/test_api_compatibility_part1.py`
- `test/legacy_test/test_batch_sampler.py`

The verifier checks the complete five-file Gold changed-file scope separately, while producing the solution patch only from the three production paths.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the existing distributed batch sampler P2P should pass and the new alias/seed tests should fail.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass after the Gold production patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

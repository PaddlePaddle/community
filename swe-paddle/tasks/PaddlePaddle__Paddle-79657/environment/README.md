# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `16037ff1effb88625041f9a1c540e8b2af3ab5c1`
- Gold commit: `20519ee630ec4929776b33341f690adc2fc48001`
- Resource: Single NVIDIA GPU
- GPU required: yes
- Build type: CUDA source build

## Critical Build Note

This task fixes a `.cu` file (CUDA kernel source). A full Paddle source build from the checkout is **required** to verify the fix. Simply installing a prebuilt wheel or overlaying Python source files is insufficient — the modified CUDA kernel must be recompiled into the Paddle runtime. After applying the gold patch, a complete `make` rebuild of the affected CUDA targets (or a full Paddle build) is mandatory before running tests.

## GPU Preflight

Before running any tests, verify that:

1. A CUDA-capable GPU is available and visible to `nvidia-smi`.
2. The CUDA toolkit version matches the Paddle build requirements for the base commit.
3. `python -c "import paddle; print(paddle.device.get_device())"` reports a valid CUDA device.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Build Paddle from source with CUDA support.
3. Apply `tests/test.patch`.
4. Run `bash tests/test.sh`; CUDA backward with `keepdim=False` and non-trailing axes should fail before the fix; forward, `keepdim=True`, trailing-axis, and CPU behavior should pass.
5. Apply `solution/code.patch`.
6. **Rebuild** the affected CUDA targets (full source rebuild recommended).
7. Run `bash tests/test.sh` again; all target tests should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Notes

- No runtime results are claimed here — all expected outcomes are described in terms of test pass/fail behavior.
- The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

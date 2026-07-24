# Environment Notes

This candidate is part of the SWE-Paddle task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `605f5e20305db0e4932a20d3e0e6cf7d7d9631d8`
- Resource: CPU + GPU
- GPU required: yes (CUDA kernels are included)
- Build path: Paddle source checkout at the base commit. This task involves C++ CPU kernels, CUDA GPU kernels, Python API, and YAML op definitions, so source build is required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the target behavior should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; the target behavior should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

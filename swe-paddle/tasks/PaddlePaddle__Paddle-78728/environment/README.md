# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `52a2dfd7f3f7f402fe13161bd6d6bce5e9a727fa`
- Resource: CPU
- GPU required: no
- Torch required: no
- CUDA required: no
- Build path: use a compatible Paddle 3.0 CPU installation and load the checkout's pure-Python Fleet recompute package.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; test collection should fail because the new public API is absent.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all original PR tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

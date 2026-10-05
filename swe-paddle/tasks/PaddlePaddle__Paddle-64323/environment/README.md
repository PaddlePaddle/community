# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `34577c4c9a14a808fd6abec145f796b30ec93c60`
- Resource: CPU
- GPU required: no
- Torch required: no
- CUDA required: no
- Build path: use a compatible Paddle 3.0 CPU installation and load the checkout's pure-Python SOT package.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Export `STRICT_MODE=1` and `MIN_GRAPH_SIZE=0`.
4. Run `bash tests/test.sh`; the target behavior should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh` again; the original PR test should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

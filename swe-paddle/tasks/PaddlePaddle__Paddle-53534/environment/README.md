# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `f74237cd73c35b8a63d7981a190a302d0ebcd03f`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. No Paddle rebuild or wheel installation is required; the verifier executes the checked-out Python implementation through AST.

The verifier uses a controlled NumPy proxy to reproduce the NumPy 1.24+ behavior where array construction for a mixed Tensor/Variable sequence raises immediately. Controlled Paddle doubles implement `assign`, `cast`, `stack`, and `squeeze` so the test validates conversion behavior without importing a historical Paddle package.

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

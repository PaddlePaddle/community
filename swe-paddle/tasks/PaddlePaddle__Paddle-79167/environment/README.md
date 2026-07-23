# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `c34031973911346f8cd98717583577f61adcf0b1`
- Resource: CPU
- GPU required: no
- Build path: Python-only checkout source with an AST overlay for the relevant top-level import/export statements; no Paddle source build is required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the new top-level API behavior should fail before the implementation.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; the target tests should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

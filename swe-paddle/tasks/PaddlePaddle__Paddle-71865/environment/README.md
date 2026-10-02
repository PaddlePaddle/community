# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `8c6c18cbe059e452bb18209f83b9dbc6c3d7d436`
- Resource: CPU
- GPU required: no
- Build path: use a compatible Paddle 3.0 CPU installation with Python 3.11 and load the checkout's pure-Python SOT package; no source rebuild is required when such an installation is available.

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

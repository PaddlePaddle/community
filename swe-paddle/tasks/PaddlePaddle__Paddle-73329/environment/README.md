# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `eb9af7cf69977981198f41ae79eebabd0e3e036f`
- Resource: CPU
- GPU required: no
- Build path: use a compatible Paddle CPU installation with Python 3.11, load the checkout's pure-Python SOT package, and export `STRICT_MODE=1` and `MIN_GRAPH_SIZE=0` as in Paddle's SOT CI.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Export `STRICT_MODE=1` and `MIN_GRAPH_SIZE=0`.
4. Run `bash tests/test.sh`; the target behavior should fail before the fix.
5. Apply `solution/code.patch`.
6. Run `bash tests/test.sh` again; the target behavior should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

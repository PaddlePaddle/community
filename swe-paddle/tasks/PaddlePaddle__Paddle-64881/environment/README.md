# Environment Notes

This candidate is part of the SWE-Paddle task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `d972f9ab8bb3d2ea5d1757a860ae45774e53b6eb`
- Resource: CPU
- Build path: Paddle source checkout at the base commit. This task only involves Python-side changes (nn functional API + Layer wrapper), so a source build with Python tests is sufficient.

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

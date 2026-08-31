# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `7b7e53fd28956700e5ed1ce68eb2aaeb59829777`
- Resource: CPU
- GPU required: no
- Build path: use an installed CPU Paddle wheel for the native runtime and load the target Python behavior from the source checkout; a Paddle source rebuild is not required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the direct Tensor and public alias cases should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

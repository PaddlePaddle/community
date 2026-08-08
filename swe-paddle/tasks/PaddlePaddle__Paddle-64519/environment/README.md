# Environment Notes

This candidate is part of the SWE-Paddle task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `2d746f9719ddd35e9e9f1330b019d996bdafbfac`
- Resource: CPU
- Build path: Paddle source checkout at the base commit. This task only involves Python-side changes (linalg API wrapping existing `paddle.linalg.inv`), so a source build with Python tests is sufficient.

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

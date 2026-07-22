# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `4f2cf7fbcaca52bb9625dc6be944f552ea1d71d5`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. A full Paddle build is not required; the verifier compiles only a lightweight C++17 harness containing the checked-out `BmmInferMeta` function.

The local cross-validation script also reports the currently installed Paddle wheel, package location, and wheel commit for environment inventory. The installed wheel is not used as the oracle for historical infermeta behavior.

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

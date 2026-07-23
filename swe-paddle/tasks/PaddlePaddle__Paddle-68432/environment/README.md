# Environment Notes

This candidate is part of the SWE-Paddle CPU10 pilot set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `979489bc3280e682f2ce8996d9b0e154ec425a59`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. For Python-only patches, the verifier may use a Python overlay; for mixed patches, source build may be required.

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

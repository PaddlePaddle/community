# Environment Notes

This candidate is part of the SWE-Paddle task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `cb5ff84d214b86b2409b3aa83ef7cd4ccd06374b`
- Resource: CPU
- Build path: Paddle source checkout at the base commit. This task only involves Python-side changes (stack APIs built on existing `concat`/`atleast_nd` primitives), so a source build with Python tests is sufficient.

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

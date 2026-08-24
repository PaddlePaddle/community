# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `b5efb98a163a2be2505e72266841e64b88254a8a`
- Resource: CPU
- GPU required: no
- Build path: Paddle source checkout at the base commit. This Python-only task executes the checkout proxy module with controlled Python modules; a Paddle native source build is not required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the target compatibility behavior should fail before the change.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

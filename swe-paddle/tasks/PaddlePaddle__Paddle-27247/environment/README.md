# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `aae41c6fca67be6a090d4f83bdf6160737d15162`
- Resource: CPU
- GPU required: no
- Build path: no source build is required; the target test executes the relevant checkout control flow through an AST overlay with controlled Python doubles.

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

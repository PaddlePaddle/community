# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `9a9345fa4dc77be655811d8e484b99cb9ff5f356`
- Resource: CPU
- GPU required: no
- Build path: no source build is required; a controlled AST overlay executes the checkout implementation with the source PR's unittest methods and local doubles for etcd and device discovery.

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

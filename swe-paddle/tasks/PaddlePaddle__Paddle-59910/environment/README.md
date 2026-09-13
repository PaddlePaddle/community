# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `5b3bc5043bcbc919a4bb9e69f421d6001814b365`
- Resource: CPU
- GPU required: no
- Build path: Python-only task; no Paddle source rebuild is required. The verifier executes checkout Python control flow through an isolated contract test.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the target PIR behavior should fail before the fix while the P2P behavior remains valid.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target behavior should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

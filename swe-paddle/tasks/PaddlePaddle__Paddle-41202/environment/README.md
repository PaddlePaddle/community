# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `23d1b3e8ed8187bfb3bd926934dd6cc71e691e53`
- Resource: CPU
- GPU required: no
- Build path: use a compatible installed CPU Paddle wheel as the runtime carrier; the verifier may overlay the checkout's target Python behavior to bridge historical module-layout differences.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the enabled auto-tune scenarios should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; the complete upstream test file should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

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
3. Run `bash tests/test.sh`; Base should collect all three tests, the disabled-mode P2P should pass, and the two enabled auto-tune F2P cases should fail because `set_autotune_config` is not present in Base.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all three benchmark tests should pass after the gold patch.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.

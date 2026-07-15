# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `6dfc086f8a3c2245ea1d75891386e82aa5721f15`
- Resource: CPU
- GPU required: no
- Patch type: Python-only
- Test framework: pytest

## Run Order

1. Check out Paddle at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the target recompute-context tests should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier should derive stable F2P and P2P node IDs from the target test file.

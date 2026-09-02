# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `0156c9d3a222adaca16a394826654a9f449d11aa`
- Resource: CPU
- GPU required: no
- Build path: use an installed Paddle wheel as the native runtime carrier and overlay only the checkout Python behavior under test; a Paddle native source build is not required.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the safe configuration loading behavior should fail before the fix.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; the complete target test file should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for deriving stable F2P and P2P node IDs from repeated runs.


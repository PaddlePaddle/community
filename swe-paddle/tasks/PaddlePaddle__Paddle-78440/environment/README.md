# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `399e85b7d6c76c49af8301f718690c8d24548554`
- Resource: CPU
- GPU required: no
- External model or network service: no
- Test framework: `pytest`
- Build requirement: Python-only patch; no C++ rebuild is required.

## Run Order

1. Check out Paddle at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the P2P case should pass and both F2P cases should fail.
4. Apply `solution/code.patch` and ensure the runtime loads the patched Python source.
5. Run `bash tests/test.sh` again; all three target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for preparing a compatible Paddle runtime and making the checkout's patched Python source visible during execution.

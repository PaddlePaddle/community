# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `199073cd2021dd05efd8b0fe79797b838f68df41`
- Resource: CPU
- GPU required: no
- Distributed multi-process execution required: no
- External model or network service: no
- Test framework: `pytest`
- Build requirement: Python-only patch; no C++ rebuild is required.

## Run Order

1. Check out Paddle at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the P2P test should pass and the F2P test should fail.
4. Apply `solution/code.patch` and ensure the patched Python source is loaded.
5. Run `bash tests/test.sh` again; both tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

The verifier is responsible for providing a compatible Paddle runtime and loading the checkout's patched Python source.

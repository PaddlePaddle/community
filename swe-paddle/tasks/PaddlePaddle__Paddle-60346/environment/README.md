# Environment Notes

This candidate is part of the SWE-Paddle community task set.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `ac2de38a80dd6f2c66b4ef4ad515e209a0470fef`
- Gold commit: `caa171552152424a2adcfe6bef9babb2039434a2`
- Resource: CPU
- GPU required: no
- Build path: Python-only task; no Paddle source rebuild is required.

The task contract test reads the checkout production files from their normal repository paths and executes the relevant Python control flow in isolation. It does not inject repository paths through `PYTHONPATH` and does not overwrite the installed Paddle package.

## Run Order

1. Check out `PaddlePaddle/Paddle` at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; one legacy P2P case should pass while the PIR F2P cases fail.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target behavior should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

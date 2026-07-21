# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `c55db2546c87e19fc78384c3497383298f3e2375`
- Resource: CPU
- GPU required: no
- Build requirement: Python-only source change; a compatible Paddle runtime with a Python source overlay is sufficient for local verification.

## Run Order

1. Check out Paddle at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the lifetime test should fail and the round-trip test should pass.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; both tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

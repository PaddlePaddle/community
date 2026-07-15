# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `406c7afec699c23158e7ff62a0f1afb306e72afe`
- Resource: CPU
- GPU required: no
- Distributed launch required: no
- Patch type: Python-only

## Run Order

1. Check out the repository at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the overlap boundary tests should fail while the non-overlap regression test passes.
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target tests should pass.

## Minimal Test Command

```bash
bash tests/test.sh
```

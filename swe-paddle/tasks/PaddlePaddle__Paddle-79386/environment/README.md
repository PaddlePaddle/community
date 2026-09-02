# Environment Notes

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `9aa3379edbee8ccd6cec772b22ad37733357f8df`
- Gold/head commit: `d9b89b3918a51476cc1755fe202f89a07f8c34d1`
- PR: `https://github.com/PaddlePaddle/Paddle/pull/79386`
- Resource: CPU
- GPU required: no
- Distributed multi-process execution required: no
- External model or network service: no
- Test framework: `pytest`
- Build requirement: source build or equivalent rebuilt `libpaddle`/pybind runtime is required for real post-fix verification.

## Run Order

1. Check out Paddle at the base commit.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh` against the base runtime. The dedicated `uint64.max` F2P assertion should fail, while the P2P guard assertions should pass.
4. Apply `solution/code.patch`.
5. Rebuild Paddle from source or provide an equivalent runtime where the patched pybind code is loaded.
6. Run `bash tests/test.sh` again. The dedicated class should pass.
7. Record the Run/Test/Fix result according to the SWE-Paddle verifier workflow.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Runtime Notes

This task changes C++ pybind code, so simply applying the patch to a source checkout while importing an existing wheel is not a valid verifier execution. The verifier must ensure Python imports the rebuilt or equivalently patched `libpaddle` corresponding to the modified checkout.

The verifier is responsible for any required source build, equivalent rebuilt runtime setup, and Run/Test/Fix execution. No Docker, GPU, or distributed execution is required by the task itself.
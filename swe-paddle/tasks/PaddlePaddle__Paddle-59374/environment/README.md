# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #59374.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `4af8ecca447eba12cf57597d95935b0b5f4311b1` (parent of #59374 merge `9fab1fe7`)
- Gold endpoint: `9fab1fe754744eaaee8c829b89bbfc9ce230ab19`
- Resource: CPU
- GPU required: no (GPU dtype/place cases are optional if CUDA is available)
- Patch type: **source build required** (C++ pybind + Python Tensor patch methods).
- Paddle install: source checkout at `base_commit`, build/install so eager Tensor methods are available.

## Run Order (Run / Test / Fix)

1. Check out base commit and complete a source build/install.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; `apply` / `apply_` cases should **fail / error** before the fix.
4. Apply `solution/code.patch` and **rebuild** (pybind changes).
5. Rerun `bash tests/test.sh`; target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- `test_apply.py` 含静态 / PIR 错误路径与 dtype 遍历；无 CUDA 时 GPU 用例应自动 skip。
- pybind 改动必须重新编译才能在 Python 侧生效。
